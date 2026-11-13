import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
import os
import json

from ddmd.inference import inference_run
from ddmd.utils import get_dir_base


class analysis_run(inference_run): 
    def __init__(self, pdb_file, md_path, ml_path) -> None:
        super().__init__(pdb_file, md_path, ml_path)
    
    def run(self, n_clusters=300, random_state=42, init='k-means++', md_form='done', 
            protein_ligand_system = "No",
            atom_sel = None,
            ligand_sel = None,
            protein_ligand_cutoff = 10.0,
            **kwargs): 
        
        vae_weight = max(self.get_trained_models(), key=os.path.getctime)
        vae_setup = os.path.join(os.path.dirname(vae_weight), "cvae.json")

        with open(vae_setup, "r") as f:
            vae_config = json.load(f)

        # ML convention:
        # atom_sel = protein selection
        # ligand_sel = ligand selection
        if atom_sel is None:
            atom_sel = vae_config.get("atom_sel", None)

        if ligand_sel is None:
            ligand_sel = vae_config.get("ligand_sel", None)
        # base analysis from infer
        if ligand_sel is not None:
            protein_ligand_system = "Yes"
        else:
            protein_ligand_system = "No"

        print("DEBUG analysis atom_sel:", atom_sel)
        print("DEBUG analysis ligand_sel:", ligand_sel)
        print("DEBUG analysis protein_ligand_system:", protein_ligand_system)

        df = self.build_md_df(form=md_form, calc_Q= False,
                              protein_ligand_system = protein_ligand_system,
                              atom_sel = atom_sel,
                              protein_sel = atom_sel,
                              ligand_sel = ligand_sel,
                              protein_ligand_cutoff = protein_ligand_cutoff, 
                              outlier_selection="com" if protein_ligand_system == "Yes" else None,
                              **kwargs)
        df['sys_label'] = [get_dir_base(i) for i in df['xtc']]
        df['gpu_id'] = [i.split('_')[2] for i in df['sys_label']]
        embeddings = np.array(df['embeddings'].to_list())

        # kmeans cluster
        df['cluster_label'] = kmeans_emb_clustering(
                embeddings, n_clusters=n_clusters, 
                init=init, random_state=random_state)
        # sort the dataframe to represent time evo
        df.sort_values(by=['gpu_id', 'sys_label', 'frame'], inplace=True)
        # save df 
        self.df = df
        df.to_pickle("result.pkl")

        # reshape dtraj
        dtrajs = {}
        for gpu_id in df.gpu_id.unique(): 
            dtraj = df[df.gpu_id == gpu_id].cluster_label.to_numpy()
            dtrajs[f'gpu_{gpu_id}'] = dtraj
        
        dtrajs['total'] = traj_stack(dtrajs.values())

        sampling_effs = []
        for run in dtrajs: 
            sampling_effs.append(
                {'run': run, 'sampling_eff': get_sampling(dtrajs[run])}
            )
        sampling_effs = pd.DataFrame(sampling_effs)
        sampling_effs.to_pickle('sampling.pkl')

        # build tree of MD simulations 


def kmeans_emb_clustering(
    embeddings, n_clusters=500, 
    random_state=42, init='k-means++'): 
    # kmeans = MiniBatchKMeans(
    #         n_clusters=n_clusters, 
    #         init=init, random_state=random_state
    #     ).fit(embeddings)
    kmeans = KMeans(
            n_clusters=n_clusters, 
            init=init, random_state=random_state, n_init= 10
        ).fit(embeddings)
    return kmeans.labels_


def traj_stack(dtrajs): 
    y_len = max(len(i) for i in dtrajs)
    dtraj_stacks = np.ones([len(dtrajs), y_len]) * -1
    for i, dtraj in enumerate(dtrajs):
        dtraj_stacks[i, :len(dtraj)] = dtraj
    return dtraj_stacks[dtraj_stacks != -1]

def get_sampling(traj): 
    sampled = []
    n_sampled = []
    for i in tqdm(traj): 
        if i not in sampled: 
            sampled.append(i)
        n_sampled.append(len(sampled))
    return n_sampled
