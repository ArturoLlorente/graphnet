import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from copy import deepcopy
from typing import Union, List, Optional
import warnings

class sensitivity_plots_jupyter:
    def __init__(self, 
                 df_original: Union[pd.DataFrame, List[pd.DataFrame]], 
                 df_labels: Union[str, List[str]],
                 db: str,
                 pulsemaps:str = 'InIceDSTPulses',
                 index_column: str = 'event_no',
                 x_pred_label: str = 'direction_x',
                 y_pred_label: str = 'direction_y',
                 z_pred_label: str = 'direction_z',
                 truth_table = 'truth'):
        
        if isinstance(df_original, pd.DataFrame):
            df_original = [df_original]
        if not isinstance(df_original, list):
            raise TypeError('df_original must be a pandas DataFrame or a list of DataFrames')
        if not isinstance(db, str):
            raise TypeError('database must be a string')
        
        self.df_original = df_original
        self.db = db
        self.pulsemaps = pulsemaps
        self.index_column = index_column
        self.x_pred_label = x_pred_label
        self.y_pred_label = y_pred_label
        self.z_pred_label = z_pred_label
        self.truth_table = truth_table
        self.df_labels = df_labels
       
    def _add_energy(self, original_df: List[pd.DataFrame]):
        df = deepcopy(original_df)
        df = [df_i.sort_values(self.index_column).reset_index(drop = True) for df_i in df]
        
        if all((df_i[self.index_column] == df[0][self.index_column]).all() for df_i in df[1:]):
            with sqlite3.connect(self.db) as con:
                query = f'select {self.index_column}, energy from {self.truth_table} where {self.index_column} in {str(tuple(df[0][self.index_column]))}'
                truth = pd.read_sql(query,con).sort_values(self.index_column).reset_index(drop = True)
        else:
            warnings.warn('The DataFrames do not have the same index', UserWarning)
            for idx, df_i in enumerate(df):
                with sqlite3.connect(self.db) as con:
                    query = f'select {self.index_column}, energy from {self.truth_table} where {self.index_column} in {str(tuple(df_i[self.index_column]))}'
                    truth = pd.read_sql(query,con).sort_values(self.index_column).reset_index(drop = True)
                    
        for idx, df_i in enumerate(df):
            for column in truth.columns:
                if column not in df_i.columns:
                    df[idx][column] = truth[column]
        return df
    
    def _add_track_label(self, original_df: List[pd.DataFrame]):
        df = deepcopy(original_df)
        df = [df_i.sort_values(self.index_column).reset_index(drop = True) for df_i in df]
        
        if all((df_i[self.index_column] == df[0][self.index_column]).all() for df_i in df[1:]):
            with sqlite3.connect(self.db) as con:
                query = f'select {self.index_column}, interaction_type, pid from {self.truth_table} where {self.index_column} in {str(tuple(df[0][self.index_column]))}'
                truth = pd.read_sql(query,con).sort_values(self.index_column).reset_index(drop = True)
                truth['track'] = 0
                truth.loc[(truth['interaction_type'] == 1) & (abs(truth['pid']) == 14), 'track'] = 1
        else:
            warnings.warn('The DataFrames do not have the same index', UserWarning)
            for idx, df_i in enumerate(df):
                with sqlite3.connect(self.db) as con:
                    query = f'select {self.index_column}, interaction_type, pid from {self.truth_table} where {self.index_column} in {str(tuple(df_i[self.index_column]))}'
                    truth = pd.read_sql(query,con).sort_values(self.index_column).reset_index(drop = True)
                    truth['track'] = 0
                    truth.loc[(truth['interaction_type'] == 1) & (abs(truth['pid']) == 14), 'track'] = 1
                            
        for idx, df_i in enumerate(df):
            for column in truth.columns:
                if column not in df_i.columns:
                    df[idx][column] = truth[column]
        
        return df

    def _add_likelihood_fit(self, original_df: pd.DataFrame, key = 'spline'):
        df = deepcopy(original_df)
        df = df.sort_values(self.index_column).reset_index(drop = True)
        if key == 'spline':
            with sqlite3.connect(self.db) as con:
                # Read the table data into a Pandas DataFrame
                query = f'SELECT {self.index_column}, zenith_spline_mpe_ic, azimuth_spline_mpe_ic from spline_mpe_ic where {self.index_column} in {str(tuple(df[self.index_column]))}'
                truth = pd.read_sql(query,con).sort_values(self.index_column).reset_index(drop = True)
                # Add the spline fit to the DataFrame
                for column in truth.columns:
                    if column not in df.columns:
                        df[column] = truth[column]    
        elif key == 'eventgen':
            with sqlite3.connect(self.db) as con:
                # Read the table data into a Pandas DataFrame
                query = f'SELECT {self.index_column}, zenith_EventGeneratorSelectedRecoNN_I3Particle, azimuth_EventGeneratorSelectedRecoNN_I3Particle from EventGeneratorSelectedRecoNN_I3Particle where {self.index_column} in {str(tuple(df[self.index_column]))}'
                truth = pd.read_sql(query,con).sort_values(self.index_column).reset_index(drop = True)
                # Add the spline fit to the DataFrame
                for column in truth.columns:
                    if column not in df.columns:
                        df[column] = truth[column]
        return df
        
    def _add_prediction_azimuth_zenith(self, original_df):
        df = deepcopy(original_df)
        df = df.sort_values(self.index_column).reset_index(drop = True)
            
        magnitude = np.sqrt(df[self.x_pred_label]**2 + df[self.y_pred_label]**2 + df[self.z_pred_label]**2)
        zenith_pred = np.arccos(df[self.z_pred_label] / magnitude)
        azimuth_pred = np.arctan2(df[self.y_pred_label], df[self.x_pred_label])
        
        zenith_pred = np.where(zenith_pred < 0, 2*np.pi + zenith_pred, zenith_pred)
        azimuth_pred = np.where(azimuth_pred < 0, 2*np.pi + azimuth_pred, azimuth_pred)
        df['zenith_pred'] = np.where(zenith_pred >= 2*np.pi, zenith_pred - 2*np.pi, zenith_pred)
        df['azimuth_pred'] = np.where(azimuth_pred >= 2*np.pi, azimuth_pred - 2*np.pi, azimuth_pred)
        
        return df


    def _calculate_percentiles(self, percentile_calculations_original, residual, key1, key2, index = None):
        percentile_calculations = deepcopy(percentile_calculations_original)
        if index is None:
            percentile_calculations[key1][key2]['p_16'].append(np.percentile(residual, 16))
            percentile_calculations[key1][key2]['p_84'].append(np.percentile(residual, 84))
            if self.include_median:
                percentile_calculations[key1][key2]['p_50'].append(np.percentile(residual, 50))
            return percentile_calculations
        else:
            if sum(index)>0:
                percentile_calculations[key1][key2]['p_16'].append(np.percentile(residual[index], 16))
                percentile_calculations[key1][key2]['p_84'].append(np.percentile(residual[index], 84))
                if self.include_median:
                    percentile_calculations[key1][key2]['p_50'].append(np.percentile(residual[index], 50))
                return percentile_calculations
            else:
                percentile_calculations[key1][key2]['p_16'].append(np.nan)
                percentile_calculations[key1][key2]['p_84'].append(np.nan)
                if self.include_median:
                    percentile_calculations[key1][key2]['p_50'].append(np.nan)
                return percentile_calculations

    def _calculate_opening_angle(self, df, key = 'gnn'):
        x = np.cos(df['azimuth']) * np.sin(df['zenith'])
        y = np.sin(df['azimuth']) * np.sin(df['zenith'])
        z = np.cos(df['zenith'])
        if key == 'gnn':
            dot_product = x*df[self.x_pred_label] + y*df[self.y_pred_label] + z*df[self.z_pred_label]
            norm = np.sqrt(df[self.x_pred_label]**2 + df[self.y_pred_label]**2 + df[self.z_pred_label]**2)
            angle = np.arccos(dot_product/norm)
        elif key == 'spline':
            x_spline = np.cos(df['azimuth_spline_mpe_ic']) * np.sin(df['zenith_spline_mpe_ic'])
            y_spline = np.sin(df['azimuth_spline_mpe_ic']) * np.sin(df['zenith_spline_mpe_ic'])
            z_spline = np.cos(df['zenith_spline_mpe_ic'])
            dot_product = x*x_spline + y*y_spline + z*z_spline
            norm = np.sqrt(x_spline**2 + y_spline**2 + z_spline**2)
            angle = np.arccos(dot_product/norm)
        elif key == 'eventgen':
            x_eventgen = np.cos(df['azimuth_EventGeneratorSelectedRecoNN_I3Particle']) * np.sin(df['zenith_EventGeneratorSelectedRecoNN_I3Particle'])
            y_eventgen = np.sin(df['azimuth_EventGeneratorSelectedRecoNN_I3Particle']) * np.sin(df['zenith_EventGeneratorSelectedRecoNN_I3Particle'])
            z_eventgen = np.cos(df['zenith_EventGeneratorSelectedRecoNN_I3Particle'])
            dot_product = x*x_eventgen + y*y_eventgen + z*z_eventgen
            norm = np.sqrt(x_eventgen**2 + y_eventgen**2 + z_eventgen**2)
            angle = np.arccos(dot_product/norm)
        else:
            assert False, "key must be 'gnn', 'spline' or 'eventgen'"
        return np.rad2deg(angle)
    
    
    def plot_resolution_fancy(self, 
                            key: str = 'direction',
                            include_median: bool = True, 
                            include_energy_hist: bool = False,
                            step: bool = True,
                            include_residual_hist:bool = False,
                            font_size: int = 10, 
                            ncols: int = 1,
                            legend_bbox: bool = None, 
                            tracks_in_dataset: bool = False,
                            cascades_in_dataset: bool = False,
                            compare_likelihood: bool = True,
                            ylims: Optional[List[int]] =  None):
        
        
        df = deepcopy(self.df_original)
        df_labels_plotting = deepcopy(self.df_labels)

        spline_idx, eventgen_idx = None, None
        if compare_likelihood:
            if tracks_in_dataset:
                spline_idx = 0
                df.insert(spline_idx, df[-1])
                df_labels_plotting.insert(spline_idx, 'SplineMPE')
            if cascades_in_dataset:
                eventgen_idx = 0 + int(tracks_in_dataset)
                df.insert(eventgen_idx, df[-1])
                df_labels_plotting.insert(eventgen_idx, 'EventGen')

        self.include_median = include_median
        
        n_lines = (len(df) + int(compare_likelihood)+1)*(int(cascades_in_dataset)+1)
        plotting_colours = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        track_index = [i for i in range(len(df))]
        track_colour = [plotting_colours[i % len(plotting_colours)] for i in track_index]
        cascade_index = [len(df)+i+1 for i in range(len(df))]
        cascade_colour = [plotting_colours[i % len(plotting_colours)] for i in cascade_index]
        all_colour = {'track': track_colour, 'cascade': cascade_colour}
        
        percentiles = ['p_16', 'p_84']
        if include_median:
            percentiles.append('p_50')
        percentile_calculations = {}
        if tracks_in_dataset:
            percentile_calculations['tracks'] = {f'{df_labels_plotting[i]}': {percentile: [] for percentile in percentiles+['mean']} for i in range(len(df))}
        if cascades_in_dataset:
            percentile_calculations['cascades'] = {f'{df_labels_plotting[i]}': {percentile: [] for percentile in percentiles+['mean']} for i in range(len(df))}
        
        fig = plt.figure(figsize = (5,3.5), constrained_layout = True)
        if include_energy_hist:
            gs = fig.add_gridspec(10, 10)
                            #left=0.1, right=0.9, bottom=0.1, top=0.9,
                            #wspace=0.05, hspace=0.01)
            ax = fig.add_subplot(gs[2:, :])
            ax_histx = fig.add_subplot(gs[0:2, :], sharex=ax)
            ax2 = fig.add_subplot(gs[2:10, 6:])
        elif include_residual_hist:
            gs = fig.add_gridspec(8, 10)
                            #left=0.1, right=0.9, bottom=0.1, top=0.9,
                            #wspace=0.05, hspace=0.01)
            ax = fig.add_subplot(gs[0:, :-4])
            ax_histx = None
            ax2 = fig.add_subplot(gs[0:8, 6:])
        else:
            gs = fig.add_gridspec(8, 6)
            ax = fig.add_subplot(gs[0:, :6])
            ax_histx = None
            ax2 = None
            for i in range(len(df)):
                if tracks_in_dataset:
                    ax.plot(np.nan, np.nan, color = all_colour['track'][i], label = f'Tracks {df_labels_plotting[i]}')
                if cascades_in_dataset:
                    ax.plot(np.nan, np.nan, color = all_colour['cascade'][i], label = f'Cascades {df_labels_plotting[i]}')
        if ylims is not None:
            ax.set_ylim(ylims[0], ylims[1])
            
            
        for i in range(len(df)):
            if tracks_in_dataset:
                percentile_calculations['tracks'][df_labels_plotting[i]]['colour'] = all_colour['track'][i]
            if cascades_in_dataset:
                percentile_calculations['cascades'][df_labels_plotting[i]]['colour'] = all_colour['cascade'][i]
        
             

        df = self._add_energy(df)
        df = self._add_track_label(df)
        df = [self._add_prediction_azimuth_zenith(df_i) if ('zenith_pred' not in df_i.columns) else df_i for df_i in df]
        if compare_likelihood:
            if tracks_in_dataset:
                df[spline_idx] = self._add_likelihood_fit(df[spline_idx], key='spline')
            if cascades_in_dataset:
                df[eventgen_idx] = self._add_likelihood_fit(df[eventgen_idx], key='eventgen')
        
        energy_bins = []# , np.arange(0,3.1,0.05)
        for percentile in np.arange(0,102.5,2.5):
            energy_bins.append(np.percentile(np.log10(df[0]['energy']), percentile))

        # Calculate the residuals and bins
        residual = {}
        if tracks_in_dataset:
            keys_df = list(percentile_calculations['tracks'].keys())
        elif cascades_in_dataset:
            keys_df = list(percentile_calculations['cascades'].keys())
          
        for idx, df_i in enumerate(df):
            key_df = keys_df[idx]
            if key == 'energy':
                if not (idx == spline_idx):
                    residual[key_df] = ((df_i['energy'] - df_i['energy_pred'])/df_i['energy']) * 100
                    key_bins = np.arange(-300,100, 5)
                    if ax2 is not None:
                        ax2.set_xlabel('$\\frac{Truth - Reco.}{Truth}$[%]', size = font_size)
                    ax.set_ylabel('Energy Resolution [%]', size = font_size)
            elif key == 'zenith':
                if idx == spline_idx:
                    residual['SplineMPE'] = (np.rad2deg(df_i['zenith'] - df_i['zenith_spline_mpe_ic']))
                elif idx == eventgen_idx:
                    residual['EventGen'] = (np.rad2deg(df_i['zenith'] - df_i['zenith_EventGeneratorSelectedRecoNN_I3Particle']))
                else:
                    residual[key_df] = np.rad2deg(df_i['zenith'] - df_i['zenith_pred'])
                key_bins = np.arange(-90, 90, 1)
                if ax2 is not None:
                    ax2.set_xlabel('Truth - Reco. [deg.]', size = font_size)
                ax.set_ylabel('Zenith Resolution [deg.]', size = font_size)
            elif key == 'azimuth':
                if idx == spline_idx:
                    residual['SplineMPE'] = (np.rad2deg(df_i['azimuth'] - df_i['azimuth_spline_mpe_ic']))
                elif idx == eventgen_idx:
                    residual['EventGen'] = (np.rad2deg(df_i['azimuth'] - df_i['azimuth_EventGeneratorSelectedRecoNN_I3Particle']))
                else:
                    residual[key_df] = np.rad2deg(df_i['azimuth'] - df_i['azimuth_pred'])
                key_bins = np.arange(-90, 90, 1)
                if ax2 is not None:
                    ax2.set_xlabel('Truth - Reco. [deg.]', size = font_size)
                ax.set_ylabel('Zenith Resolution [deg.]', size = font_size)
            elif key == 'direction':
                if idx == spline_idx:
                    residual['SplineMPE'] = self._calculate_opening_angle(df_i, key = 'spline')
                elif idx == eventgen_idx:
                    residual['EventGen'] = self._calculate_opening_angle(df_i, key = 'eventgen')
                else:
                    residual[key_df] = self._calculate_opening_angle(df_i, key = 'gnn')
                key_bins = np.arange(0, 120, 1)
                if ax2 is not None:
                    ax2.set_xlabel('Opening Angle [deg.]', size = font_size)
                ax.set_ylabel('Opening Angle [deg.]', size = font_size)
            else:
                assert False, "key must be 'energy', 'zenith', 'azimuth' or 'direction'"

        for idx, df_i in enumerate(df):
            for i in range(len(energy_bins) - 1):
                index_energy = (np.log10(df_i['energy'])>= energy_bins[i]) & (np.log10(df_i['energy'])<energy_bins[i+1])                    
                for event_type in percentile_calculations.keys():
                    index = np.where(event_type == 'tracks', df_i['track'][index_energy] == 1, df_i['track'][index_energy] == 0)
                    percentile_calculations = self._calculate_percentiles(percentile_calculations, residual[df_labels_plotting[idx]][index_energy], event_type, df_labels_plotting[idx], index = index)
                    percentile_calculations[event_type][df_labels_plotting[idx]]['mean'].append(np.log10(df_i['energy'][index_energy][index]).mean())               
        
        for event_type in list(percentile_calculations.keys()):
            for pred_name in list(percentile_calculations[event_type].keys()):
                for percentile in percentiles+['mean']:
                    percentile_calculations[event_type][pred_name][percentile] = np.array(percentile_calculations[event_type][pred_name][percentile])
                    
                    
                    
        for event_type in list(percentile_calculations.keys()):
            for pred_name in list(percentile_calculations[event_type].keys()):
                for percentile in percentiles:
                    ls = '-' if percentile == 'p_50' else '--'
                    if step:
                        if ls != '--':
                            ax.step(10**percentile_calculations[event_type][pred_name]['mean'], percentile_calculations[event_type][pred_name][percentile], label=None, ls=ls, color=percentile_calculations[event_type][pred_name]['colour'], where='mid')
                    else:
                        ax.plot(10**percentile_calculations[event_type][pred_name]['mean'], percentile_calculations[event_type][pred_name][percentile], label=None, ls=ls, color=percentile_calculations[event_type][pred_name]['colour'])
                    if ls == '--':
                        ax.fill_between(10**percentile_calculations[event_type][pred_name]['mean'], percentile_calculations[event_type][pred_name]['p_16'], percentile_calculations[event_type][pred_name]['p_84'], alpha=0.1, color=percentile_calculations[event_type][pred_name]['colour'])

        ax.plot(np.nan, np.nan, label = '84th', ls = '--', color = 'grey')
        if include_median:
            ax.plot(np.nan, np.nan, label = 'Median', ls = '-', color = 'grey')
        
        if include_energy_hist:
            for idx, df_i in enumerate(df):
                ax_histx.hist(np.log10(df_i['energy'][df_i['track'] == 1]), histtype = 'step', color = track_colour[idx], bins = energy_bins)

        ax.legend(frameon = False, fontsize = font_size, ncol = ncols, bbox_to_anchor=legend_bbox)
        ax.set_xlabel('True Energy [GeV]', size = font_size)
        if ax2 is not None:
            for idx, df_i in enumerate(df):
                key_df = keys_df[idx]
                if tracks_in_dataset:
                    ax2.hist(residual[df['track'] == 1], histtype = 'step', bins = key_bins, label = 'Tracks')
                if cascades_in_dataset:
                    ax2.hist(residual[df['track'] == 0], histtype = 'step', bins = key_bins, label = 'Cascades')
            if tracks_in_dataset:
                ax2.hist(residual[df['track'] == 1], histtype = 'step', bins = key_bins, label = 'Tracks')
            if cascades_in_dataset:
                ax2.hist(residual[df['track'] == 0], histtype = 'step', bins = key_bins, label = 'Cascades')
            if spline_idx is not None:
                ax2.hist(residual[df['track'] == 1], histtype = 'step', bins = key_bins, label = 'Tracks (SplineMPE)')
            if eventgen_idx is not None:
                ax2.hist(residual[df['track'] == 0], histtype = 'step', bins = key_bins, label = 'Cascades (EventGen)')
        
        if key in ['energy', 'zenith', 'azimuth']:
            if ax2 is not None:
                ax2.legend(frameon = False, fontsize = font_size, loc = 'upper left')
        else:
            if ax2 is not None:
                ax2.legend(frameon = False, fontsize = font_size)
    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if include_energy_hist:
            ax_histx.spines['right'].set_visible(False)
            ax_histx.spines['top'].set_visible(False)
            ax_histx.spines['left'].set_visible(False)
            ax_histx.set_yticklabels([])
            plt.setp(ax_histx.get_xticklabels(), visible=False)
            ax_histx.set_xticklabels([])
            ax_histx.set_yticks([])
        if ax2 is not None:
            ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.yaxis.tick_right()
        #plt.suptitle(f'{key.capitalize()}', size = font_size)
        ax.set_xscale('log')

        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10 
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        #plt.text(x = text_loc[0], y = text_loc[1], s = "IceCube Simulation", fontsize = 12)
        fig.savefig(f'{key}_reco.png')

        return None


if __name__ == '__main__':
    df = [pd.read_csv('/scratch/users/allorana/prediction_cascades_icemix/model1_base.csv'),
          pd.read_csv('/scratch/users/allorana/prediction_cascades_icemix/model2_base.csv'),
          pd.read_csv('/scratch/users/allorana/prediction_cascades_icemix/model3_base.csv'),
          pd.read_csv('/scratch/users/allorana/prediction_cascades_icemix/model4_base.csv'),
          pd.read_csv('/scratch/users/allorana/prediction_cascades_icemix/model5_base.csv')]
    db = '/scratch/users/allorana/cascades_21537.db'
    df_labels = ['model1', 'model2', 'model3', 'model4', 'model5']

    sp = sensitivity_plots_jupyter(df, df_labels, db, x_pred_label = 'direction_y', y_pred_label = 'direction_x', z_pred_label = 'direction_z')

    #x_pred_label, y_pred_label, z_pred_label = 'direction_y', 'direction_x', 'direction_z'

    ax = sp.plot_resolution_fancy(key = 'direction',
                                include_median = True,
                                include_energy_hist = False,
                                step = True,
                                include_residual_hist = False,
                                cascades_in_dataset = True,
                                compare_likelihood = True,
                                ylims = [5, 30])