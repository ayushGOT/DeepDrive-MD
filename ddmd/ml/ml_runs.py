import glob
import json
import os
import time
from functools import reduce 
from multiprocessing import Process
from operator import mul
from typing import List, Optional

import MDAnalysis as mda
import numpy as np
import tensorflow as tf
from MDAnalysis.analysis import distances
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ddmd.utils import (
    build_logger,
    create_path,
    get_numoflines,
    separate_kwargs,
    yml_base,
    touch_file,
)

from .model_vae import CVAE

logger = build_logger()


class ml_base(yml_base):
    """
    Run ML training

    Parameters
    ----------
    pdb_file : ``str``
        Coordinate file, can also use topology file

    md_path : ``str``
        Path of MD simulations, where all the simulation information
        is stored

    n_train_start : ``int``
        Number of frame to start training
    """

    def __init__(
        self,
        pdb_file,
        md_path,
    ) -> None:
        super().__init__()
        print(pdb_file)
        if type(pdb_file) is list:                         # if multiple ICs
            pdb_file = (pdb_file)[0]
            self.pdb_file = os.path.abspath(pdb_file)
        else:                                       # only 1 IC
            self.pdb_file = os.path.abspath(pdb_file)
        self.md_path = md_path

    ### -> new function
    def get_available_md_runs(self):
        run_dirs = glob.glob(f"{self.md_path}/md_run_*")
        available_dirs = []
    
        for run_dir in run_dirs:
            if not os.path.isdir(run_dir):
                continue
    
            trained_file = os.path.join(run_dir, "TRAINED")
            log_file = os.path.join(run_dir, "output.log")
            xtc_file = os.path.join(run_dir, "output.xtc")
    
            if (
                not os.path.exists(trained_file)
                and os.path.exists(log_file)
                and os.path.exists(xtc_file)
            ):
                available_dirs.append(run_dir)

        logger.info(f"Available MD runs (not TRAINED): {len(available_dirs)}")
        for d in available_dirs:
            logger.info(f"  Using MD folder: {d}")

        return available_dirs

    
    ### -> modified function
    def get_numberofFrames(self):
        """
        This method assumes log and traj are with the same output
        frequency
        """
        available_dirs = self.get_available_md_runs()
        log_files = []

        for run_dir in available_dirs:
            log_file = os.path.join(run_dir, "output.log")
            if os.path.exists(log_file):
                log_files.append(log_file)

        total_frames = sum(get_numoflines(log) - 1 for log in log_files)
        logger.info(f"Total frames available for training: {total_frames}")
        return total_frames
        # log_files = sorted(glob.glob(f"{self.md_path}/md_run_*/*.log"))
        # return sum(get_numoflines(log) - 1 for log in log_files)

    def get_contact_maps(
        self,
        atom_sel: str = "name CA",
        ligand_sel: Optional[str] = None,
        map_type: str= "binary",
        cutoff: float = 8.0,
        dry_run: bool = False,
    ):
        # only use one traj file if dry run
        # if dry_run:
        #     xtc_files = sorted(glob.glob(f"{self.md_path}/md_run_*/*.xtc"))[:1]
        # else:
        #     xtc_files = sorted(glob.glob(f"{self.md_path}/md_run_*/*.xtc"))

        if dry_run:
        # For dry_run, we only need one valid trajectory to determine CVAE input shape.
        # It does NOT need to be untrained.
            logger.info(f"if part is running, dry_run {dry_run}")
            # done_files = sorted(
            #     glob.glob(f"{self.md_path}/md_run_*/DONE"),
            #     key=os.path.getctime
            # )

            # xtc_files = []
            # for done_file in done_files:
            #     run_dir = os.path.dirname(done_file)
            #     xtc_file = os.path.join(run_dir, "output.xtc")
            #     if os.path.exists(xtc_file):
            #         xtc_files.append(xtc_file)
            xtc_files = sorted(
                glob.glob(f"{self.md_path}/md_run_*/output.xtc"),
                key=os.path.getctime
            )

            xtc_files = xtc_files[:1]
        else:
            logger.info(f"Else block is running, dry_run {dry_run}")
            # For real training, only use completed-but-untrained MD runs.
            available_dirs = self.get_available_md_runs()
            xtc_files = []

            for run_dir in available_dirs:
                xtc_file = os.path.join(run_dir, "output.xtc")
                if os.path.exists(xtc_file):
                    xtc_files.append(xtc_file)

        xtc_files = sorted(xtc_files, key=os.path.getctime)

        # only use one traj file if dry run
        if dry_run:
            xtc_files = xtc_files[:1]

        logger.info(f"Collecting cm for CVAE. Atom sel={atom_sel}")
        logger.info(f"Ligand sel={ligand_sel}")
        cm_list = []


        for xtc in tqdm(xtc_files):
            try:
                mda_u = mda.Universe(self.pdb_file, xtc)
            except:
                logger.debug(f"Skipping {xtc}...")
                continue

            # ------------------------------------------------------------
            # CASE 1: original behavior
            # If ligand_sel is not provided, self-contact map
            # uses only atom_sel
            # ------------------------------------------------------------

            if ligand_sel is None:
                ca = mda_u.select_atoms(atom_sel)
                
                for _ in mda_u.trajectory:  
                    if map_type == "binary":
                        cm = (distances.self_distance_array(ca.positions) < cutoff) * 1.0
                    elif map_type == "distance":
                        cm = distances.self_distance_array(ca.positions)
                        cm[cm > cutoff] = 50.0  # not interested in extremely long-range interactions
                    cm_list.append(cm)

            
            # ------------------------------------------------------------
            # CASE 2: protein-ligand contact map
            # If ligand_sel is provided, uses atom_sel as protein selection
            # and ligand_sel as ligand selection
            # ------------------------------------------------------------


# protein_ligand_system------start

            else:
                protein_atoms = mda_u.select_atoms(atom_sel)
                ligand_atoms = mda_u.select_atoms(ligand_sel)

                logger.info(f"Selected protein atoms: {len(protein_atoms)}")
                logger.info(f"Selected ligand atoms: {len(ligand_atoms)}")

                if len(protein_atoms) == 0:
                    raise ValueError(f"atom_sel selected 0 atoms: {atom_sel}")

                if len(ligand_atoms) == 0:
                    raise ValueError(f"ligand_sel selected 0 atoms: {ligand_sel}")
                
                for ts in mda_u.trajectory:
                    if map_type == "binary":
                        cm = (
                            distances.distance_array(
                                protein_atoms.positions,
                                ligand_atoms.positions,
                                box=ts.dimensions
                            ) < cutoff
                        ) * 1.0

                    elif map_type == "distance":
                        cm = distances.distance_array(
                            protein_atoms.positions,
                            ligand_atoms.positions,
                            box=ts.dimensions
                        )
                        cm[cm > cutoff] = 50.0

                    cm_list.append(cm)

# protein_ligand_system------end
            
            
        
        if map_type == "distance":
            max_dist = np.max(cm_list)
            cm_list = [cm / max_dist for cm in cm_list]   # normalize all distances to (0,1)
        
        return np.array(cm_list)

    def get_vae_input(self, cm_list: Optional[List] = None, padding: int = 2, **kwargs):
        if cm_list is None:
            cm_list = self.get_contact_maps(**kwargs)
        logger.debug(f"  Padding {padding} on the contact maps...")
        cvae_input = cm_to_cvae(cm_list, padding=padding)
        logger.debug(f"cvae input shape: {cvae_input.shape}")
        return cvae_input

    def get_padding(self, strides):
        """calculate padding for vae input"""
        padding = reduce(mul, [i[0] for i in strides])
        padding = max(2, padding)
        return padding

    def build_vae(
        self,
        latent_dim=3,
        n_conv_layers=4,
        feature_maps=[16, 16, 16, 16],
        filter_shapes=[[3, 3], [3, 3], [3, 3], [3, 3]],
        strides=[[1, 1], [1, 1], [1, 1], [1, 1]],
        dense_layers=1,
        dense_neurons=[128],
        dense_dropouts=[0.3],
        **kwargs,
    ):
        input_kwargs, kwargs = separate_kwargs(self.get_contact_maps, kwargs)
        padding = self.get_padding(strides)
        cvae_input = self.get_vae_input(padding=padding, **input_kwargs)
        
        image_size = cvae_input.shape[1:-1]
        channel = cvae_input.shape[-1]
        
        cvae = CVAE(
            image_size,
            channel,
            n_conv_layers,
            feature_maps,
            filter_shapes,
            strides,
            dense_layers,
            dense_neurons,
            dense_dropouts,
            latent_dim,
            **kwargs,
        )
        return cvae, cvae_input

    def train_cvae(self, retrain_lvl=0, batch_size=256, epochs=100, **kwargs):

        prev_weight = None   #####
        available_dirs = self.get_available_md_runs()

        logger.info(f"Training on {len(available_dirs)} MD folders:")
        for d in available_dirs:
            logger.info(f"  Training data from: {d}")

        if len(available_dirs) == 0:
            logger.info("No available untrained MD folders found.")
            return None, kwargs


        if retrain_lvl > 1:   ### -> started modifying here
            ml_run_dir = os.path.abspath(os.path.join(self.md_path, "..", "ml_run"))
            prev_pattern = os.path.join(
                ml_run_dir, f"vae_run_retrain_{retrain_lvl-1:03}_*"
            )
            prev_dirs = sorted(glob.glob(prev_pattern))

            if prev_dirs:
                prev_dir = max(prev_dirs, key=os.path.getctime)   # take the latest matching folder
                prev_weight = os.path.join(prev_dir, "cvae_weight.h5")

        cvae, cvae_input = self.build_vae(**kwargs)

        logger.info(f"Total training samples (frames): {len(cvae_input)}")

        if len(cvae_input) == 0:  ### -> started modifying here
           logger.info("No new frames found for training; skipping this retrain.")
           return None, kwargs

        if prev_weight and os.path.exists(prev_weight):
            logger.info(f"Loading previous weights from {prev_weight}")
            cvae.load(prev_weight) ### -> ended modifying here


        train_data, val_data = data_split(cvae_input)
        cvae.train(train_data, batch_size, epochs=epochs, validation_data=val_data)
        # cvae, cvae_setup = self.train_cvae(**kwargs)
        save_path = create_path(sys_label=f"retrain_{retrain_lvl:03}", dir_type="vae")
        with open(f"{save_path}/cvae.json", "w") as json_file:
            json.dump(kwargs, json_file)
        cvae.save(f"{save_path}/cvae_weight.h5")

        saved_weight = f"{save_path}/cvae_weight.h5"
        if not os.path.exists(saved_weight):
            raise RuntimeError(f"Model save failed: {saved_weight}")
        
        for run_dir in available_dirs:
            done_file = os.path.join(run_dir, "DONE")
            trained_file = os.path.join(run_dir, "TRAINED")

            if os.path.exists(done_file):
                touch_file(trained_file)
                logger.info(f"Marked as TRAINED: {run_dir}")
            else:
                logger.info(
                    f"Training completed, but folder is not DONE yet; "
                    f"leaving unmarked: {run_dir}"
                )

        return cvae, kwargs


class ml_run(ml_base):
    def __init__(self, pdb_file, md_path, n_train_start=1000) -> None:
        super().__init__(pdb_file, md_path)
        self.n_train_start = n_train_start

    # def ddmd_run(self, retrain_freq=2000, **kwargs):
    #     retrain_lvl = 0
    #     ###last_trained_frame = kwargs.get("last_trained_frame", 0)
    #     ###chunk_size = kwargs.get("chunk_size", 1000)
    #     while True:
    #         # decide whether to start training
    #         n_frames = self.get_numberofFrames()
    #         if n_frames - last_trained_frame < chunk_size:  ### i changed here
    #             # logger.debug(f" Collected {n_frames} out of "\
    #             #         f"{self.n_train_start} frames for training")
    #             continue
    #         else:
    #             self.n_train_start = (
    #                 n_frames + retrain_freq
    #                 if retrain_freq > 2
    #                 else n_frames * retrain_freq
    #             )
    #             retrain_lvl += 1
    #             logger.info(f"Starting training with {n_frames} frames...")

    #         kwargs["retrain_lvl"] = retrain_lvl
    #         ### kwargs["last_trained_frame"] = last_trained_frame
    #         ml_p = Process(target=self.train_cvae, kwargs=kwargs)
    #         ml_p.start()
    #         ml_p.join()
    #         ###last_trained_frame = n_frames

    #         # cvae, cvae_setup = self.train_cvae(**kwargs)
    #         # save_path = create_path(sys_label=f'retrain_{retrain_lvl:03}',
    #         #                     dir_type='vae')
    #         # with open(f"{save_path}/cvae.json", 'w') as json_file:
    #         #     json.dump(cvae_setup, json_file)
    #         # cvae.save(f"{save_path}/cvae_weight.h5")
    #         logger.info(
    #             f"  Finished training, next training will "
    #             f"start with {self.n_train_start} frames..."
    #         )

    ### ddmd_run -> modified
    def ddmd_run(self, retrain_freq=500, **kwargs):
        logger.info("Waiting 60 seconds before checking existing ML runs...")
        time.sleep(60)
        retrain_lvl = self.get_latest_retrain_lvl()
        logger.info(f"Starting ML retrain counter from existing latest level: {retrain_lvl}")

        last_training_frame_count = 0
    
        while True:
            n_frames = self.get_numberofFrames()
    
            # first training uses n_train_start, later trainings use retrain_freq
            threshold = self.n_train_start if retrain_lvl == 0 else retrain_freq

            new_available_frames = n_frames - last_training_frame_count

            has_done_folder = any(
            os.path.exists(os.path.join(run_dir, "DONE"))
            for run_dir in self.get_available_md_runs()
        )

            logger.info(
            f"Available frames: {n_frames}; "
            f"frames added since last training: {new_available_frames}; "
            f"threshold: {threshold}"
            )
    
            # if new_available_frames < threshold:
            #     logger.info(f"Not enough frames yet. {new_available_frames}/{threshold}). Sleeping for 60s")
            #     time.sleep(60)
            #     continue
            if new_available_frames < threshold and not (
                has_done_folder and new_available_frames > 0
            ):
                logger.info(
                    f"Not enough frames yet "
                    f"({new_available_frames}/{threshold}). Sleeping for 60s"
                )
                time.sleep(60)
                continue

            if has_done_folder and 0 < new_available_frames < threshold:
                logger.info(
                    f"DONE folder detected with {new_available_frames} remaining frames. "
                    f"Running final training even though threshold is {threshold}."
                )
    
            retrain_lvl += 1
            logger.info(
                f"Starting training with {n_frames} currently available frames... "
                f"(threshold={threshold})..."
            )
    
            kwargs["retrain_lvl"] = retrain_lvl
    
            ml_p = Process(target=self.train_cvae, kwargs=kwargs)
            ml_p.start()
            ml_p.join()
    
            if ml_p.exitcode == 0:
                logger.info("Finished training successfully.")

                last_training_frame_count = self.get_numberofFrames()

                logger.info(
                    f"Reset available-frame baseline to "
                    f"{last_training_frame_count}"
                )
            else:
                logger.error(
                    f"Training process failed with exit code {ml_p.exitcode}. "
                    f"Will retry the MD folders."
                )

    def get_latest_retrain_lvl(self):
        ml_run_dir = os.path.abspath(os.path.join(self.md_path, "..", "ml_run"))

        logger.info(f"Searching previous ML runs in: {ml_run_dir}")

        retrain_dirs = glob.glob(os.path.join(ml_run_dir, "vae_run_retrain_*"))

        logger.info(f"Found previous retrain directories: {retrain_dirs}")

        latest_lvl = 0

        for d in retrain_dirs:
                base = os.path.basename(d)
                # expected format: vae_run_retrain_232_1779620029
                parts = base.split("_")
                try:
                    lvl = int(parts[3])
                    logger.info(f"Found retrain level {lvl} from {base}")
                    latest_lvl = max(latest_lvl, lvl)
                except:
                    continue

        logger.info(f"Latest retrain level detected: {latest_lvl}")


        return latest_lvl


def cm_to_cvae(cm_data, padding=2):
    """
    A function converting the 2d upper triangle information of contact maps
    read from hdf5 file to full contact map and reshape to the format ready
    for cvae
    """
    # transfer upper triangle to full matrix
    
    cm_data = np.asarray(cm_data)

    # Original behavior:
    # self_distance_array gives one upper-triangle vector per frame.
    if cm_data.ndim == 2:
        cm_data_full = np.array([triu_to_full(cm) for cm in cm_data])

    # New behavior:
    # protein-ligand distance_array already gives rectangular matrix per frame.
    elif cm_data.ndim == 3:
        cm_data_full = cm_data

    else:
        raise ValueError(f"Unexpected contact map shape: {cm_data.shape}")

    # padding if odd dimension occurs in image
    pad_f = lambda x: (0, 0) if x % padding == 0 else (0, padding - x % padding)
    padding_buffer = [(0, 0)]
    for x in cm_data_full.shape[1:]:
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode="constant")

    # reshape matrix to 4d tensor
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))
    
    return cvae_input


def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full


def data_split(x, train_size=0.7, test_size=False):
    train, val = train_test_split(x, train_size=train_size)
    if test_size:
        val, test_data = train_test_split(val, train_size=test_size)
        return train, val, test_data
    else:
        return train, val

