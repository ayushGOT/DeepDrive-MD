import os
import glob
import json
import time
import pandas as pd
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

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

def native_contacts(ref_pdb):    # to determine atom pairs to be used for considering native contacts
    mda_ref= mda.Universe(ref_pdb)

    # Define 3' and 5' strand residues
    three_prime_residues = [8, 9, 10]
    five_prime_residues = [1, 2, 3]
    
    # Define nucleobase atoms (excluding sugar/phosphate)
    nucleobase_atoms = ['C2', 'C4', 'C5', 'C6', 'C8', 'N1', 'N3', 'N7', 'N9', 'O2', 'N2', 'O6', 'N6', 'O4']  # includes purine + pyrimidine variants
    
    q1 = []
    q2 = []
    ref = []
    
    # Loop over all inter-strand nucleotide pairs
    for res_i in three_prime_residues:
        for res_j in five_prime_residues:
            # Select heavy nucleobase atoms for each residue
            # atoms_i = mda_ref.select_atoms(f"resid {res_i} and nucleicbase")  # this didn't work for residues 1 and 10
            # atoms_j = mda_ref.select_atoms(f"resid {res_j} and nucleicbase")
            atoms_i = mda_ref.select_atoms(
                f"resid {res_i} and name {' '.join(nucleobase_atoms)} and not name H*"
            )
            atoms_j = mda_ref.select_atoms(
                f"resid {res_j} and name {' '.join(nucleobase_atoms)} and not name H*"
            )
    
            # Loop over all atom pairs
            for ai in atoms_i:
                for aj in atoms_j:
                    dist = np.linalg.norm(ai.position - aj.position)
                    if dist < 5.0:  # Å
                        q1.append(ai.index + 1)  # convert to 1-based index
                        q2.append(aj.index + 1)
                        ref.append(dist * 0.1 * 1.5)  # convert to nm and scale by 1.5
    logger.info("Determined atom pairs for considering native contacts.")
    return q1,q2,ref

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
            return sorted(glob.glob(f'{self.md_path}/md_run*/*xtc'))
        elif form.lower() == 'done': 
            md_done = sorted(glob.glob(f'{self.md_path}/md_run*/DONE'))
            return [f'{os.path.dirname(i)}/output.xtc' for i in md_done]
        elif form.lower() == 'running': 
            return [i for i in self.get_md_runs(form='all') if i not in self.get_md_runs(form='done')]
        else: 
            raise("Form not defined, using all, done or running ...")

    def build_md_df(self, ref_pdb=None, atom_sel="name CA", form='all', calc_Q=False, compute_shap=False, shap_batch_size=100, shap_outdir="shap_maps", xtc_old=None, **kwargs): 
        xtc_all = self.get_md_runs(form=form)
        if xtc_old:  # if the DDMD workflow was continued and a df already exists for the older xtcs
            xtc_files = list(set(xtc_all) - set(xtc_old))
        df_entry = []
        if ref_pdb: 
            ref_u = mda.Universe(ref_pdb)
            sel_ref = ref_u.select_atoms(atom_sel)    # atom selection to calculate RMSD
        vae_config = self.get_cvae(dry_run=True)
        cm_sel = vae_config['atom_sel'] if 'atom_sel' in vae_config else 'name CA'   # atom selection for contact maps
        cm_cutoff = vae_config['cutoff'] if 'cutoff' in vae_config else 8
        map_type = vae_config['map_type'] if 'map_type' in vae_config else 'binary'
        logger.info(f"Processing MD trajectories using {map_type} contact maps")
        cm_list = []
        
        ## Extracting atom pairs for considering native contacts (Q)
        if calc_Q:
            q1,q2,ref= native_contacts(ref_pdb)
            
        for xtc in tqdm(xtc_files): 
            setup_yml = f"{os.path.dirname(xtc)}/setting.yml"
            pdb_file = dict_from_yaml(setup_yml)['pdb_file']
            try: 
                mda_u = mda.Universe(self.pdb_file, xtc)
            except: 
                logger.info(f"Skipping {xtc}...")
                continue

            sel_atoms = mda_u.select_atoms(atom_sel)  # atom selection to calculate RMSD
            sel_cm = mda_u.select_atoms(cm_sel)
            for ts in mda_u.trajectory:  
                if map_type == "binary":
                    cm = (distances.self_distance_array(sel_cm.positions) < cm_cutoff) * 1.0
                elif map_type == "distance":
                    cm = distances.self_distance_array(sel_cm.positions)
                    cm[cm > cm_cutoff] = 50.0                # not interested in extremely long-range interactions
                cm_list.append(cm)
                local_entry = {'pdb': os.path.abspath(pdb_file), 
                            'xtc': os.path.abspath(xtc), 
                            'frame': ts.frame}
                if ref_pdb: 
                    rmsd = rms.rmsd(
                            sel_atoms.positions, sel_ref.positions, 
                            superposition=True)
                    local_entry['rmsd'] = rmsd
                
                ## Calculating and storing Q
                if calc_Q:
                    r= np.array([])
                    for i in range(len(ref)):
                        atom1, atom2 =mda_u.select_atoms("bynum %i" %q1[i]), mda_u.select_atoms("bynum %i" %q2[i])
                        dist= distances.distance_array(atom1.positions,atom2.positions)*0.1
                        r=np.append(r,dist)
                    Q= contacts.soft_cut_q(r, ref, beta=50.0, lambda_constant=1.0)
                    local_entry['Q'] = Q
                df_entry.append(local_entry)
                
        if map_type == "distance":
            max_dist = max(max(sublist) for sublist in cm_list)
            cm_list = [[x / max_dist for x in sublist] for sublist in cm_list]   # normalize all distances to (0,1)
        
        df = pd.DataFrame(df_entry)
        if 'strides' in vae_config: 
            padding = self.get_padding(vae_config['strides'])
        vae_input = self.get_vae_input(cm_list, padding=padding)
        embeddings = self.vae.return_embeddings(vae_input)
        df['embeddings'] = embeddings.tolist()
        outlier_score = lof_score_from_embeddings(embeddings, **kwargs)
        
        for i, _ in enumerate(outlier_score):   # in case we want to control the outlierness
            outlier_score[i]= outlier_score[i] if outlier_score[i] > -100 else 0
        df['lof_score'] = outlier_score
    
        # SHAP computation
        if compute_shap:
            import shap
            from tensorflow.keras.models import Model
            
            if not os.path.exists(shap_outdir):
                os.makedirs(shap_outdir)

            shap_values_sum = None
            sample_count = 0
            #print(f'Model: {self.vae.embedder}')
            #self.vae.embedder.summary()
            print(f'shape of model output: {self.vae.embedder.output.shape}')
            vae_input = vae_input[::200]   # skip every 100 frames; uncomment if you don't wanna consider all frames
            background = vae_input[np.random.choice(len(vae_input), size=200, replace=False)]
            print(f'shape of vae input: {(vae_input).shape}')
            np.save(os.path.join(shap_outdir, f"vae_input.npy"), vae_input)
            
            explainer = shap.DeepExplainer(self.vae.embedder, background)
            #encoder_model = Model(inputs=self.vae.input, outputs=self.vae.embedder.get_layer("dense_1").output)
            #explainer = shap.DeepExplainer(encoder_model, background)

            # Accumulate SHAP values across all batches
            all_shap_values = []
            for i in range(0, len(vae_input), shap_batch_size):
                batch = vae_input[i:i+shap_batch_size]
                shap_vals = explainer.shap_values(batch)  # single ndarray: (batch, 20, 20, 1, latent_dim)
                shap_vals = np.array(shap_vals)
                if isinstance(shap_vals, list):
                    shap_vals = np.array(shap_vals)
                all_shap_values.append(shap_vals)

            shap_values = np.concatenate(all_shap_values, axis=0)  # shape: (samples, 20, 20, 1, latent_dim)
            print(f"Final SHAP shape: {shap_values.shape}")
            np.save(os.path.join(shap_outdir, f"shap_values.npy"), shap_values)

            num_latent_dims = shap_values.shape[-1]
            avg_shap_maps = []
            for i in range(num_latent_dims):
                shap_i = shap_values[..., i]  # shape: (samples, 20, 20, 1)
                shap_i = np.squeeze(shap_i)   # remove singleton dim
                if shap_i.ndim == 4:
                    shap_i = np.mean(shap_i, axis=0)  # shape: (20, 20)
                elif shap_i.ndim == 3:
                    shap_i = np.mean(shap_i, axis=0)  # shape: (20, 20)
                else:
                    raise ValueError("Unexpected shape after slicing SHAP values")
                avg_shap_maps.append(shap_i)

            self.avg_shap_maps = avg_shap_maps
            logger.info("Computed average SHAP maps for each latent dimension.")

            for dim, shap_map in enumerate(avg_shap_maps):
                np.save(os.path.join(shap_outdir, f"shap_map_latent_dim_{dim}.npy"), shap_map)
            logger.info(f"Saved SHAP maps to directory: {shap_outdir}")
            
            def plot_shap_beeswarms(vae_input_path, shap_values_path, outdir="shap_beeswarms"):
                """
                Plots a SHAP beeswarm plot for each latent dimension.

                Parameters:
                    vae_input_path (str): Path to the saved VAE input (shape: N x 20 x 20 x 1).
                    shap_values_path (str): Path to saved SHAP values (shape: N x 20 x 20 x 1 x ld).
                    outdir (str): Directory where beeswarm plots will be saved.
                """

                # Load data
                #vae_input = np.load(vae_input_path)                    # (N, 20, 20, 1)
                #shap_values = np.load(shap_values_path)                # (N, 20, 20, 1, latent_dim)
                N, h, w, c, D = shap_values.shape

                # Flatten inputs and SHAP values
                X_flat = vae_input.reshape(N, -1)                      # (N, 400)
                contact_labels = [f"c{i}" for i in range(X_flat.shape[1])]

                os.makedirs(outdir, exist_ok=True)

                for d in range(D):
                    print(f"Plotting beeswarm for latent dimension {d}...")
                    shap_flat = shap_values[..., d].reshape(N, -1)     # (N, 400)

                    # Create Explanation object
                    explainer = shap.Explainer(lambda x: x, X_flat)    # dummy model
                    shap_exp = shap.Explanation(values=shap_flat, data=X_flat, feature_names=contact_labels)

                    # Plot and save
                    plt.figure(figsize=(5,5))
                    shap.plots.beeswarm(shap_exp, max_display=30, show=False)
                    out_path = os.path.join(outdir, f"beeswarm_latent_dim_{d}.png")
                    plt.title(f"SHAP Beeswarm - Latent Dim {d}")
                    plt.tight_layout()
                    plt.savefig(out_path, dpi=300)
                    plt.close()
                    print(f"Saved: {out_path}")

            plot_shap_beeswarms(vae_input, shap_values, outdir="shap_beeswarms")

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
        time.sleep(30)          # to allow time for writing the h5 file before it's opened
        logger.info(f" ML nn loaded weight from {vae_label}")
        self.vae.load(vae_weight)
        return vae_config

    def ddmd_run(self, n_outliers=50, 
            md_threshold=0.75, screen_iter=10, nudge='no', restart_pdb=None, target=None, 
                 lower_bound=None, upper_bound=None, **kwargs): 
        
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
                    get_numoflines(md_done[0].replace('xtc', 'log')) - 1
            # build the dataframe and rank outliers 
            df = self.build_md_df(**kwargs)
            logger.info(f"Built dataframe from {len(df)} frames. ")
            if 'ref_pdb' in kwargs:
                logger.info(f"Lowest RMSD: {min(df['rmsd'])} A, "\
                    f"Highest RMSD: {max(df['rmsd'])} A. " )
            
            if lower_bound and upper_bound:   # to apply restraint on outliers wrt a physical quantity
                df_outliers = df[(df['rmsd'] >= lower_bound) & (df['rmsd'] <= upper_bound)]
                logger.info(f"Restricting outlier RMSD within {lower_bound} and {upper_bound} A ...")
            elif upper_bound:
                df_outliers = df[(df['rmsd'] <= upper_bound)]
                logger.info(f"Restricting outlier RMSD < {upper_bound} A ...")
            elif lower_bound:
                df_outliers = df[(df['rmsd'] >= lower_bound)]
                logger.info(f"Restricting outlier RMSD > {lower_bound} A ...")
            else:
                df_outliers = df
            
            if df_outliers.empty:
                logger.info(f"no frame sampled in the specified RMSD limit. ")
                continue

            if len(df_outliers) < n_outliers:
                df_outliers = df_outliers.sort_values('lof_score')
            else:
                df_outliers = df_outliers.sort_values('lof_score').head(n_outliers)

            if 'ref_pdb' in kwargs: 
                if nudge == 'high':     # nudge towards higher RMSD
                    df_outliers = df_outliers.sort_values(by='rmsd', ascending=False)
                    logger.info("nudging towards higher RMSD")
                if nudge == 'low':      # nudge towards lower RMSD
                    df_outliers = df_outliers.sort_values(by='rmsd', ascending=True)
                    logger.info("nudging towards lower RMSD")
            
            if iteration % screen_iter == 0: 
                logger.info(df_outliers.to_string())
                save_df = f"df_outlier_iter_{iteration}.pkl"
                df_outliers.to_pickle(save_df)
            # assess simulations 
            sim_running = self.get_md_runs(form='running')
            sim_to_stop = [i for i in sim_running \
                    if i not in set(df_outliers['xtc'].to_list())]
            # only stop simulations that have been running for a while
            # 3/4 done
            sim_to_stop = [i for i in sim_to_stop \
                    if get_numoflines(i.replace('xtc', 'log')) \
                    > len_md_done * md_threshold]
            for i, sim in enumerate(sim_to_stop): 
                sim_path = os.path.dirname(sim)
                sim_run_len = get_numoflines(sim.replace('xtc', 'log'))
                if sim_run_len >= len_md_done: 
                    logger.info(f"{get_dir_base(sim)} finished before inferencing, skipping...")
                    continue
                restart_frame = f"{sim_path}/new_pdb"
                if os.path.exists(restart_frame): 
                    continue
                outlier = df_outliers.iloc[i]
                outlier.to_json(restart_frame)
                logger.info(f"{get_dir_base(sim)} finished "\
                    f"{get_numoflines(sim.replace('xtc', 'log'))} "\
                    f"of {len_md_done} frames, yet no outlier detected.")
                logger.info(f"Writing new pdb from frame "\
                    f"{outlier['frame']} of {get_dir_base(outlier['xtc'])} "\
                    f"to {get_dir_base(sim)}")
                # write_pdb_frame(, xtc, frame, save_path=None)

            logger.info(f"\n=======> Done iteration {iteration} <========\n")
            time.sleep(1)
            iteration += 1
        

def lof_score_from_embeddings(
            embeddings, n_neighbors=20, **kwargs):
    clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,**kwargs).fit(embeddings) 
    return clf.negative_outlier_factor_

