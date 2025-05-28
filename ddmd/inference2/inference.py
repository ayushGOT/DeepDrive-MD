import os
import glob
import json
import time
import pandas as pd
import MDAnalysis as mda
import numpy as np

from typing import List
from MDAnalysis.analysis import rms
from MDAnalysis.analysis import distances
from MDAnalysis.analysis import contacts
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from ddmd.ml import ml_base
from ddmd.utils import build_logger, dict_from_yaml, get_numoflines, get_dir_base

logger = build_logger()

## for Q calculation ###

q1= [15.,16.,16.,16.,16.,17.,17.,17.,17.,17.,17.,18.,18.,18.,18.,18.,20.,20.,
 20.,20.,21.,21.,21.,21.,49.,49.,49.,50.,50.,50.,50.,50.,50.,50.,50.,50.,
 51.,51.,51.,51.,51.,51.,51.,51.,51.,51.,51.,51.,52.,52.,52.,52.,52.,52.,
 52.,52.,52.,52.,52.,52.,52.,52.,54.,54.,54.,54.,54.,54.,54.,54.,54.,55.,
 55.,55.,55.,55.,55.,55.,55.,55.,55.,58.,58.,82.,84.,84.,84.,84.,84.,84.,
 84.,85.,85.,85.,85.,85.,85.,85.,88.,88.,88.,88.,88.,88.,88.,88.,89.,89.,
 89.,89.,89.,90.,90.,90.]    # to store atom no. 1 from the pair of atoms

q2= [315.,314.,315.,318.,319.,284.,312.,314.,315.,318.,319.,314.,315.,318.,
 319.,320.,314.,318.,319.,320.,314.,318.,319.,320.,284.,315.,318.,250.,
 283.,284.,287.,288.,314.,315.,318.,319.,249.,250.,251.,281.,283.,284.,
 287.,288.,312.,314.,315.,318.,250.,251.,283.,284.,287.,288.,289.,309.,
 312.,314.,315.,318.,319.,320.,283.,284.,287.,288.,289.,314.,318.,319.,
 320.,254.,283.,287.,288.,289.,309.,314.,318.,319.,320.,318.,320.,250.,
 249.,250.,251.,253.,254.,284.,287.,248.,249.,250.,251.,283.,284.,287.,
 249.,250.,251.,253.,254.,283.,284.,287.,249.,250.,251.,253.,254.,251.,
 253.,254.]      # to store atom no. 2 from the pair of atoms

ref= [0.744,0.621,0.539,0.553,0.721,0.733,0.743,0.536,0.408,0.528,0.722,0.564,
 0.551,0.436,0.565,0.565,0.736,0.577,0.65 ,0.589,0.747,0.56 ,0.569,0.462,
 0.745,0.737,0.703,0.713,0.618,0.538,0.553,0.72 ,0.622,0.579,0.602,0.73 ,
 0.713,0.628,0.736,0.742,0.534,0.408,0.529,0.723,0.742,0.592,0.495,0.623,
 0.737,0.689,0.563,0.552,0.437,0.567,0.566,0.738,0.695,0.589,0.606,0.564,
 0.645,0.709,0.734,0.749,0.575,0.649,0.589,0.71 ,0.639,0.662,0.68 ,0.729,
 0.746,0.56 ,0.569,0.462,0.704,0.749,0.689,0.664,0.686,0.739,0.73 ,0.741,
 0.619,0.534,0.564,0.736,0.747,0.717,0.732,0.746,0.538,0.408,0.551,0.717,
 0.61 ,0.716,0.551,0.527,0.435,0.575,0.56 ,0.731,0.715,0.664,0.72 ,0.722,
 0.566,0.65 ,0.57 ,0.564,0.587,0.461]  # to store the native distances (already mutiplied by 1.5) b/w the pair of atoms

#####

class inference_run(ml_base): 
    """
    Inferencing between MD and ML

    Parameters
    ----------
    pdb_file : ``str``
        Coordinate file, can also use topology file

    md_path : ``str`` 
        Path of MD simulations, where all the simulation information
        is stored

    ml_path : ``str``
        Path of VAE or other ML traning directory, used to search 
        trained models
    """
    def __init__(self, 
        pdb_file, 
        md_path,
        ml_path,
        ) -> None:
        super().__init__(pdb_file, md_path)
        self.ml_path = ml_path
        self.vae = None
        # self.outlier_path = create_path(dir_type='inference')

    def get_trained_models(self): 
        return sorted(glob.glob(f"{self.ml_path}/vae_run_*/*h5"))

    def get_md_runs(self, form:str='all') -> List: 
        if form.lower() == 'all': 
            return sorted(glob.glob(f'{self.md_path}/md_run*/*output.dcd'))
        elif form.lower() == 'done': 
            md_done = sorted(glob.glob(f'{self.md_path}/md_run*/DONE'))
            return [f'{os.path.dirname(i)}/output.dcd' for i in md_done]
        elif form.lower() == 'running': 
            return [i for i in self.get_md_runs(form='all') if i not in self.get_md_runs(form='done')]
        else: 
            raise("Form not defined, using all, done or running ...")

    def build_md_df(self, ref_pdb=None, atom_sel="name CA", form='all', **kwargs): 
        dcd_files = self.get_md_runs(form=form)
        df_entry = []
        if ref_pdb: 
            ref_u = mda.Universe(ref_pdb)
            sel_ref = ref_u.select_atoms(atom_sel)
        vae_config = self.get_cvae(dry_run=True)
        cm_sel = vae_config['atom_sel'] if 'atom_sel' in vae_config else 'name CA'
        cm_cutoff = vae_config['cutoff'] if 'cutoff' in vae_config else 8
        logger.info("Processing MD trajectories.")
        cm_list = []
        for dcd in tqdm(dcd_files): 
            setup_yml = f"{os.path.dirname(dcd)}/setting.yml"
            pdb_file = dict_from_yaml(setup_yml)['pdb_file']
            try: 
                mda_u = mda.Universe(self.pdb_file, dcd)
            except: 
                logger.info(f"Skipping {dcd}...")
                continue

            sel_atoms = mda_u.select_atoms(atom_sel)
            sel_cm = mda_u.select_atoms(cm_sel)
            for ts in mda_u.trajectory: 
                cm = (distances.self_distance_array(sel_cm.positions) < cm_cutoff) * 1.0
                cm_list.append(cm)
                local_entry = {'pdb': os.path.abspath(pdb_file), 
                            'dcd': os.path.abspath(dcd), 
                            'frame': ts.frame}
                if ref_pdb: 
                    rmsd = rms.rmsd(
                            sel_atoms.positions, sel_ref.positions, 
                            superposition=True)
                    local_entry['rmsd'] = rmsd
                # possible new analysis
                ## Calculating and storing Q
                r= np.array([])
                for i in range(114):
                    atom1, atom2 =mda_u.select_atoms("bynum %i" %q1[i]), mda_u.select_atoms("bynum %i" %q2[i])
                    dist= distances.distance_array(atom1.positions,atom2.positions)*0.1
                    r=np.append(r,dist)
                Q= contacts.soft_cut_q(r, ref, beta=50.0, lambda_constant=1.0)
                local_entry['Q'] = Q
                df_entry.append(local_entry)
        
        df = pd.DataFrame(df_entry)
        if 'strides' in vae_config: 
            padding = self.get_padding(vae_config['strides'])
        vae_input = self.get_vae_input(cm_list, padding=padding)
        embeddings = self.vae.return_embeddings(vae_input)
        df['embeddings'] = embeddings.tolist()
        outlier_score = lof_score_from_embeddings(embeddings, **kwargs)
#        for i, _ in enumerate(outlier_score):
 #           outlier_score[i]= outlier_score[i] if outlier_score[i] > -100 else 0
        df['lof_score'] = outlier_score
        return df

    def get_cvae(self, **kwargs): 
        """
        getting the last model so far
        """
        # get weight 
        vae_weight =  max(self.get_trained_models(), key=os.path.getctime)  # to select the latest CVAE model
        vae_setup = os.path.join(os.path.dirname(vae_weight), 'cvae.json')
        vae_label = os.path.basename(os.path.dirname(vae_weight))
        # get conf 
        vae_config = json.load(open(vae_setup, 'r'))
        if self.vae is None: 
            self.vae, _ = self.build_vae(**vae_config, **kwargs)
            logger.info(f"vae created from {vae_setup}")
        # load weight
        logger.info(f" ML nn loaded weight from {vae_label}")
        self.vae.load(vae_weight)
        return vae_config

    def ddmd_run(self, n_outliers=50, 
            md_threshold=0.75, screen_iter=10, nudge='no', **kwargs): 
        iteration = 0
        while True: 
            trained_models = self.get_trained_models() 
            if trained_models == []: 
                continue
            md_done = self.get_md_runs(form='done')
            if md_done == []: 
                continue
            else: 
                len_md_done = \
                    get_numoflines(md_done[0].replace('dcd', 'log')) - 1
            # build the dataframe and rank outliers 
            df = self.build_md_df(**kwargs)
            logger.info(f"Built dataframe from {len(df)} frames. ")
            df_outliers = df.sort_values('lof_score').head(n_outliers)
          #  if iteration % 2 ==0:
           #     df_outliers = df_outliers.sample(frac = 1)  # shuffling the outliers which were already sorted based on LOF scores

            if 'ref_pdb' in kwargs: 
                #df_outliers = df_outliers.sort_values(by='Q', ascending=True)
                logger.info(f"Lowest RMSD: {min(df['rmsd'])} A, "\
                    f"lowest outlier RMSD: {min(df_outliers['rmsd'])} A. "\
                    f"Highest RMSD: {max(df['rmsd'])} A. " \
                    f"Lowest Q: {min(df['Q'])} , "\
                    f"Highest Q: {max(df['Q'])} ")
            # 
            if iteration % screen_iter == 0: 
                logger.info(df_outliers.to_string())
                save_df = f"df_outlier_iter_{iteration}.pkl"
                df_outliers.to_pickle(save_df)
            # assess simulations 
            sim_running = self.get_md_runs(form='running')
            sim_to_stop = [i for i in sim_running \
                    if i not in set(df_outliers['dcd'].to_list())]
            # only stop simulations that have been running for a while
            # 3/4 done
            sim_to_stop = [i for i in sim_to_stop \
                    if get_numoflines(i.replace('dcd', 'log')) \
                    > len_md_done * md_threshold]
            for i, sim in enumerate(sim_to_stop): 
                sim_path = os.path.dirname(sim)
                sim_run_len = get_numoflines(sim.replace('dcd', 'log'))
                if sim_run_len >= len_md_done: 
                    logger.info(f"{get_dir_base(sim)} finished before inferencing, skipping...")
                    continue
                restart_frame = f"{sim_path}/new_pdb"
                if os.path.exists(restart_frame): 
                    continue
                outlier = df_outliers.iloc[i]
                outlier.to_json(restart_frame)
                logger.info(f"{get_dir_base(sim)} finished "\
                    f"{get_numoflines(sim.replace('dcd', 'log'))} "\
                    f"of {len_md_done} frames, yet no outlier detected.")
                logger.info(f"Writing new pdb from frame "\
                    f"{outlier['frame']} of {get_dir_base(outlier['dcd'])} "\
                    f"to {get_dir_base(sim)}")
                # write_pdb_frame(, dcd, frame, save_path=None)

            logger.info(f"\n=======> Done iteration {iteration} <========\n")
            time.sleep(1)
            iteration += 1
        

def lof_score_from_embeddings(
            embeddings, n_neighbors=20, **kwargs):
    clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,**kwargs).fit(embeddings) 
    return clf.negative_outlier_factor_

