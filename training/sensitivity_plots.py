import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from copy import deepcopy

class sensitivity_plots:

    def __init__(self, 
                 df_original: pd.DataFrame, 
                 db: str,
                 pulsemaps:str = 'InIceDSTPulses',
                 index_column: str = 'event_no',
                 x_pred_label: str = 'direction_x',
                 y_pred_label: str = 'direction_y',
                 z_pred_label: str = 'direction_z',
                 truth_table = 'truth'):
        
        if not isinstance(df_original, pd.DataFrame):
            raise TypeError('df_original must be a pandas DataFrame')
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

    def _add_energy(self, original_df, db):
        df = deepcopy(original_df)
        df = df.sort_values(self.index_column).reset_index(drop = True)
        with sqlite3.connect(db) as con:
            query = f'select {self.index_column}, energy from {self.truth_table} where {self.index_column} in {str(tuple(df[self.index_column]))}'
            truth = pd.read_sql(query,con).sort_values(self.index_column).reset_index(drop = True)
        
        for column in truth.columns:
            if column not in df.columns:
                df[column] = truth[column]
        return df

    def _add_track_label(self, original_df, db):
        df = deepcopy(original_df)
        df = df.sort_values(self.index_column).reset_index(drop = True)
        with sqlite3.connect(db) as con:
            query = f'select {self.index_column}, interaction_type, pid from {self.truth_table} where {self.index_column} in {str(tuple(df[self.index_column]))}'
            truth = pd.read_sql(query,con).sort_values(self.index_column).reset_index(drop = True)
        
        truth['track'] = 0
        truth.loc[(truth['interaction_type'] == 1) & (abs(truth['pid']) == 14), 'track'] = 1

        for column in truth.columns:
            if column not in df.columns:
                df[column] = truth[column]
        return df

    def _add_spline_fit(self, original_df, db):
        df = deepcopy(original_df)
        df = df.sort_values(self.index_column).reset_index(drop = True)
        with sqlite3.connect(db) as con:
            # Read the table data into a Pandas DataFrame
            query = f'SELECT {self.index_column}, zenith_spline_mpe_ic, azimuth_spline_mpe_ic from spline_mpe_ic where {self.index_column} in {str(tuple(df[self.index_column]))}'
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


    def _calculate_percentiles(self, residual, index = None):
        if index is None:
            return np.percentile(residual, 16),np.percentile(residual, 84), np.percentile(residual[index], 50)
        else:
            if sum(index)>0:
                return np.percentile(residual[index], 16),np.percentile(residual[index], 84), np.percentile(residual[index], 50)
            else:
                return np.nan, np.nan, np.nan

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
        return np.rad2deg(angle)

    def plot_2d_performance_plot(self, key, fill = True, font_size = 10):
        df = self.df_original
        fig = plt.figure(figsize = (5,3.5), constrained_layout = True)
        ax = fig.add_subplot(111)
        if key == 'energy':
            key_bins = np.arange(0,3.5, 0.1)
            ax.set_xlabel('True Energy [GeV]', size = font_size)
            ax.set_ylabel('Reco. Energy [GeV]', size = font_size)

        if key == 'zenith':
            key_bins = np.arange(-1, 1, 0.05)
            ax.set_xlabel('True  cos(zenith) ', size = font_size)
            ax.set_ylabel('Reco. cos(zenith) ', size = font_size)
            
        if key == 'azimuth':
            key_bins = np.arange(-1, 1, 0.05)
            ax.set_xlabel('True  cos(azimuth) ', size = font_size)
            ax.set_ylabel('Reco. cos(azimuth) ', size = font_size)
        
        pct_16 = []
        pct_84 = []
        pct_50 = []
        means = []
        medians = []
        for k in range(len(key_bins)-1):
            if key == 'energy':
                idx = (np.log10(df[key]) >= key_bins[k]) & (np.log10(df[key])< key_bins[k +1])
                if sum(idx)> 0:
                    pct_16.append(np.percentile(np.log10(df[key+ '_pred'][idx]),16))
                    pct_50.append(np.percentile(np.log10(df[key+ '_pred'][idx]),50))
                    pct_84.append(np.percentile(np.log10(df[key+ '_pred'][idx]),84))
                    means.append(np.mean(np.log10(df[key][idx])))
                    medians.append(np.median(np.log10(df[key][idx])))
            elif key == 'zenith' or key == 'azimuth':
                idx = (np.cos(df[key]) >= key_bins[k]) & (np.cos(df[key])< key_bins[k +1])
                if sum(idx)>0:
                    pct_16.append(np.percentile(np.cos(df[key+ '_pred'][idx]),16))
                    pct_50.append(np.percentile(np.cos(df[key+ '_pred'][idx]),50))
                    pct_84.append(np.percentile(np.cos(df[key+ '_pred'][idx]),84))
                    means.append(np.mean(np.cos(df[key][idx])))
                    medians.append(np.median(np.cos(df[key][idx])))
        
        if key == 'energy':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.plot(df[key],df[key], label = '1:1', ls = '--', color = 'grey', alpha = 0.5)
            if fill:
                ax.fill_between(10**np.array(means), 10**np.array(pct_16), 10**np.array(pct_84), color = 'tab:blue', alpha = 0.5, label = 'Central 68%')
            #ax.plot(10**np.array(means), 10**np.array(pct_16), label = None, ls = '--', color = 'tab:blue', alpha = 0.8)
            #ax.plot(medians, pct_50, label = '50th', ls = '-', color = 'tab:blue', alpha = 0.8)
            #ax.plot(10**np.array(means), 10**np.array(pct_84), label = None, ls = '-.', color = 'tab:blue', alpha = 0.8)
            ax.plot(10**np.array(means), 10**np.array(pct_50), label = 'Median', ls = '-', color = 'tab:blue', alpha = 1)
            
        if key == 'zenith':
            ax.plot(np.cos(df[key]),np.cos(df[key]), label = '1:1', ls = '--', color = 'grey', alpha = 0.5)
            if fill:
                ax.fill_between(x = means, y1 = pct_84, y2 = pct_16, color = 'tab:blue', alpha = 0.5, label = 'Central 68%')
        
            ax.plot(means, pct_16, label = None, ls = '--', color = 'tab:blue', alpha = 0.8)
            ax.plot(means, pct_84, label = None, ls = '-.', color = 'tab:blue', alpha = 0.8)
            
            ax.plot(means, pct_50, label = 'Median', ls = '-', color = 'tab:blue', alpha = 1)
            ax.plot(medians, pct_50, label = '50th', ls = '-', color = 'tab:blue', alpha = 0.8)
            
        if key == 'azimuth':
            ax.plot(np.cos(df[key]),np.cos(df[key]), label = '1:1', ls = '--', color = 'grey', alpha = 0.5)
            if fill:
                ax.fill_between(x = means, y1 = pct_84, y2 = pct_16, color = 'tab:blue', alpha = 0.5, label = 'Central 68%')
        
            ax.plot(means, pct_16, label = None, ls = '--', color = 'tab:blue', alpha = 0.8)
            ax.plot(means, pct_84, label = None, ls = '-.', color = 'tab:blue', alpha = 0.8)
            
            ax.plot(means, pct_50, label = 'Median', ls = '-', color = 'tab:blue', alpha = 1)
        ax.legend(frameon = False, fontsize = font_size)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #plt.text(x = text_loc[0], y = text_loc[1], s = "IceCube Simulation", fontsize = 12)
        #plt.rcParams['xtick.labelsize'] = 10
        #plt.rcParams['ytick.labelsize'] = 10 
        #plt.rcParams['axes.labelsize'] = 10
        #plt.rcParams['axes.titlesize'] = 12
        #plt.rcParams['legend.fontsize'] = 10
        #fig.savefig(f'2d_correlation_performance{key}.pdf')
        #fig.savefig(f'2d_correlation_performance{key}.png')
    def plot_resolution_fancy(self, 
                            key,
                            pulse_count_threshold = 8, 
                            include_median = True, 
                            include_energy_hist = False,
                            step = True,
                            include_residual_hist = False,
                            font_size =10, 
                            text_loc = 'center',
                            ncols = 1,
                            legend_bbox = None, 
                            cascades_in_dataset = False,
                            compare_spline = True):
        #font_size = 15
        df = self.df_original
        plotting_colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        track_colour = plotting_colours[0]
        if cascades_in_dataset:
            cascade_colour = plotting_colours[1]
            if compare_spline:
                track_spline_colour = plotting_colours[2]
                cascade_spline_colour = plotting_colours[3]
        else:
            if compare_spline:
                track_spline_colour = plotting_colours[1]
            
        
        fig = plt.figure(figsize = (5,3.5), constrained_layout = True)
        if include_energy_hist:
            gs = fig.add_gridspec(10, 10)
                            #left=0.1, right=0.9, bottom=0.1, top=0.9,
                            #wspace=0.05, hspace=0.01)
            ax = fig.add_subplot(gs[2:, :-4])
            ax_histx = fig.add_subplot(gs[0:2, :-4], sharex=ax)
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
                            #left=0.1, right=0.9, bottom=0.1, top=0.9,
                            #wspace=0.05, hspace=0.01)
            ax = fig.add_subplot(gs[0:, :6])
            ax_histx = None
            ax2 = None
            ax.plot(np.nan, np.nan, color = track_colour, label = 'Tracks (GNN)')
            if compare_spline:
                ax.plot(np.nan, np.nan, color = track_spline_colour, label = 'Tracks (SplineMPE)')
            if cascades_in_dataset:
                ax.plot(np.nan, np.nan, color = cascade_colour, label = 'Cascades (GNN)')
                if compare_spline:
                    ax.plot(np.nan, np.nan, color = cascade_spline_colour, label = 'Cascades (SplineMPE)')

        if 'energy' not in df.columns:
            df = self._add_energy(df, self.db)
        if 'track' not in df.columns:
            df = self._add_track_label(df, self.db)
        if 'zenith_pred' not in df.columns:
            df = self._add_prediction_azimuth_zenith(df)
        if compare_spline:
            if 'zenith_spline_mpe_ic' not in df.columns:
                df = self._add_spline_fit(df, self.db)
        
        energy_bins = []# , np.arange(0,3.1,0.05)
        for percentile in np.arange(0,105,5):
            energy_bins.append(np.percentile(np.log10(df['energy']), percentile))
        mean_energy_track = []
        mean_energy_cascade = []
        mean_energy_track_spline = []
        mean_energy_cascade_spline = []
        
        p_16 = {'track': [],
                'cascade': []}
        p_84 = {'track': [],
                'cascade': []}
        p_50 = {'track': [],
                'cascade': []}
        if compare_spline:
            p_16['track_splinempe'] = []
            p_84['track_splinempe'] = []
            p_50['track_splinempe'] = []
            p_16['cascade_splinempe'] = []
            p_84['cascade_splinempe'] = []
            p_50['cascade_splinempe'] = []

        if key == 'energy':
            residual = ((df['energy'] - df['energy_pred'])/df['energy']) * 100
            key_bins = np.arange(-300,100, 5)
            if ax2 is not None:
                ax2.set_xlabel('$\\frac{Truth - Reco.}{Truth}$[%]', size = font_size)
            ax.set_ylabel('Energy Resolution [%]', size = font_size)
        elif key == 'zenith':
            residual = np.rad2deg(df['zenith'] - df['zenith_pred'])
            if compare_spline:
                residual_spline = np.rad2deg(df['zenith'] - df['zenith_spline_mpe_ic'])
            key_bins = np.arange(-90, 90, 1)
            if ax2 is not None:
                ax2.set_xlabel('Truth - Reco. [deg.]', size = font_size)
            ax.set_ylabel('Zenith Resolution [deg.]', size = font_size)
        elif key == 'direction':
            residual = self._calculate_opening_angle(df, key = 'gnn')
            if compare_spline:
                residual_spline = self._calculate_opening_angle(df, key = 'spline')
            key_bins = np.arange(0, 120, 1)
            if ax2 is not None:
                ax2.set_xlabel('Opening Angle [deg.]', size = font_size)
            ax.set_ylabel('Opening Angle [deg.]', size = font_size)
        else:
            assert False, "key must be 'energy', 'zenith' or 'direction'"

        for i in range(len(energy_bins) - 1):
            index_energy = (np.log10(df['energy'])>= energy_bins[i]) & (np.log10(df['energy'])<energy_bins[i+1])
            index_track = df['track'][index_energy] == 1
            if cascades_in_dataset:
                index_cascade = df['track'][index_energy] == 0
            
            p_16_track, p_84_track, p_50_track = self._calculate_percentiles(residual[index_energy], index = index_track)
            mean_energy_track.append(np.log10(df['energy'][index_energy][index_track]).mean())
            p_16['track'].append(p_16_track)
            p_84['track'].append(p_84_track)
            p_50['track'].append(p_50_track)
            
            if cascades_in_dataset:
                p_16_cascade, p_84_cascade, p_50_cascade = self._calculate_percentiles(residual[index_energy], index = index_cascade)
                mean_energy_cascade.append(np.log10(df['energy'][index_energy][index_cascade]).mean())
                p_16['cascade'].append(p_16_cascade)
                p_84['cascade'].append(p_84_cascade)
                p_50['cascade'].append(p_50_cascade)
                
            if compare_spline:
                p_16_track_spline, p_84_track_spline, p_50_track_spline = self._calculate_percentiles(residual_spline[index_energy], index = index_track)
                p_16['track_splinempe'].append(p_16_track_spline)
                p_84['track_splinempe'].append(p_84_track_spline)
                p_50['track_splinempe'].append(p_50_track_spline)
                if cascades_in_dataset:
                    p_16_cascade_spline, p_84_cascade_spline, p_50_cascade_spline = self._calculate_percentiles(residual_spline[index_energy], index = index_cascade)
                    p_16['cascade_splinempe'].append(p_16_cascade_spline)
                    p_84['cascade_splinempe'].append(p_84_cascade_spline)
                    p_50['cascade_splinempe'].append(p_50_cascade_spline)
            
        p_16['track'] = np.array(p_16['track'])
        p_84['track'] = np.array(p_84['track'])
        p_50['track'] = np.array(p_50['track'])
        mean_energy_track = np.array(mean_energy_track)
        if cascades_in_dataset:
            p_16['cascade'] = np.array(p_16['cascade'])
            p_84['cascade'] = np.array(p_84['cascade'])
            p_50['cascade'] = np.array(p_50['cascade'])
            mean_energy_cascade = np.array(mean_energy_cascade)
            
        if compare_spline:
            p_16['track_splinempe'] = np.array(p_16['track_splinempe'])
            p_84['track_splinempe'] = np.array(p_84['track_splinempe'])
            p_50['track_splinempe'] = np.array(p_50['track_splinempe'])
            mean_energy_track_spline = np.array(mean_energy_track)
            if cascades_in_dataset:
                p_16['cascade_splinempe'] = np.array(p_16['cascade_splinempe'])
                p_84['cascade_splinempe'] = np.array(p_84['cascade_splinempe'])
                p_50['cascade_splinempe'] = np.array(p_50['cascade_splinempe'])
                mean_energy_cascade_spline = np.array(mean_energy_cascade)
        
        if step:
            ax.step(10**mean_energy_track, p_16['track'], label = None, ls = '--', color = track_colour, where = 'mid')
            ax.step(10**mean_energy_track, p_84['track'], label = None, ls = '--', color = track_colour, where = 'mid')
            if cascades_in_dataset:
                ax.step(10**mean_energy_cascade, p_16['cascade'], label = None, ls = '--', color = cascade_colour, where = 'mid')
                ax.step(10**mean_energy_cascade, p_84['cascade'], label = None, ls = '--', color = cascade_colour, where = 'mid')
            if compare_spline:
                ax.step(10**mean_energy_track_spline, p_16['track_splinempe'], label = None, ls = '--', color = track_spline_colour, where = 'mid')
                ax.step(10**mean_energy_track_spline, p_84['track_splinempe'], label = None, ls = '--', color = track_spline_colour, where = 'mid')
                if cascades_in_dataset:
                    ax.step(10**mean_energy_cascade_spline, p_16['cascade_splinempe'], label = None, ls = '--', color = cascade_spline_colour, where = 'mid')
                    ax.step(10**mean_energy_cascade_spline, p_84['cascade_splinempe'], label = None, ls = '--', color = cascade_spline_colour, where = 'mid')
            if include_median:
                ax.step(10**mean_energy_track, p_50['track'], label = None, ls = '-', color = track_colour, where = 'mid')
                if cascades_in_dataset:
                    ax.step(10**mean_energy_cascade, p_50['cascade'], label = None, ls = '-', color = cascade_colour, where = 'mid')
                if compare_spline:
                    ax.step(10**mean_energy_track_spline, p_50['track_splinempe'], label = None, ls = '-', color = track_spline_colour, where = 'mid')
                    if cascades_in_dataset:
                        ax.step(10**mean_energy_cascade_spline, p_50['cascade_splinempe'], label = None, ls = '-', color = cascade_spline_colour, where = 'mid')
        
        else:
            ax.plot(10**mean_energy_track, p_16['track'], label = None, ls = '--', color = track_colour)
            ax.plot(10**mean_energy_track, p_84['track'], label = None, ls = '-', color = track_colour)
            if cascades_in_dataset:
                ax.plot(10**mean_energy_cascade, p_16['cascade'], label = None, ls = '--', color = cascade_colour)
                ax.plot(10**mean_energy_cascade, p_84['cascade'], label = None, ls = '-', color = cascade_colour)
            if compare_spline:
                ax.plot(10**mean_energy_track_spline, p_16['track_splinempe'], label = None, ls = '--', color = track_spline_colour)
                ax.plot(10**mean_energy_track_spline, p_84['track_splinempe'], label = None, ls = '-', color = track_spline_colour)
                if cascades_in_dataset:
                    ax.plot(10**mean_energy_cascade_spline, p_16['cascade_splinempe'], label = None, ls = '--', color = cascade_spline_colour)
                    ax.plot(10**mean_energy_cascade_spline, p_84['cascade_splinempe'], label = None, ls = '-', color = cascade_spline_colour)
            if include_median:
                ax.plot(10**mean_energy_track, p_50['track'], label = None, ls = '-', color = track_colour)
                if cascades_in_dataset:
                    ax.plot(10**mean_energy_cascade, p_50['cascade'], label = None, ls = '-', color = cascade_colour)
                if compare_spline:
                    ax.plot(10**mean_energy_track_spline, p_50['track_splinempe'], label = None, ls = '-', color = track_spline_colour)
                    if cascades_in_dataset:
                        ax.plot(10**mean_energy_cascade_spline, p_50['cascade_splinempe'], label = None, ls = '-', color = cascade_spline_colour)
            
        ax.plot(np.nan, np.nan, label = '84th', ls = '--', color = 'grey')
        if include_median:
            ax.plot(np.nan, np.nan, label = 'Median', ls = '-', color = 'grey')
        #ax.plot(np.nan, np.nan, label = 'Central 68%', ls = '--', color = 'grey')
        
        if include_energy_hist:
            ax_histx.hist(np.log10(df['energy'][df['track'] == 1]), histtype = 'step', color = track_colour, bins = energy_bins)
            if compare_spline:
                ax_histx.hist(np.log10(df['energy'][df['track'] == 1]), histtype = 'step', color = track_spline_colour, bins = energy_bins)  
            if cascades_in_dataset:
                ax_histx.hist(np.log10(df['energy'][df['track'] == 0]), histtype = 'step', color = cascade_colour, bins = energy_bins)
                if compare_spline:
                    ax_histx.hist(np.log10(df['energy'][df['track'] == 0]), histtype = 'step', color = cascade_spline_colour, bins = energy_bins)

        ax.legend(frameon = False, fontsize = font_size, ncol = ncols, bbox_to_anchor=legend_bbox)
        ax.set_xlabel('True Energy [GeV]', size = font_size)
        if ax2 is not None:
            ax2.hist(residual[df['track'] == 1], histtype = 'step', bins = key_bins, label = 'Tracks')
            if cascades_in_dataset:
                ax2.hist(residual[df['track'] == 0], histtype = 'step', bins = key_bins, label = 'Cascades')
            if compare_spline:
                ax2.hist(residual[df['track'] == 1], histtype = 'step', bins = key_bins, label = 'Tracks (SplineMPE)')
                if cascades_in_dataset:
                    ax2.hist(residual[df['track'] == 0], histtype = 'step', bins = key_bins, label = 'Cascades (SplineMPE)')
        
        if key in ['energy', 'zenith']:
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
        #fig.savefig(f'{key}_reco.pdf')
        fig.savefig(f'{key}_reco.png')
        if ax2 is not None:
            return ax, ax2
        else:
            return ax

