#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import all of the python packages used in this workflow.
import scipy
import numpy as np
from collections import OrderedDict
import os, sys
from pylab import *
import pandas as pd
import numpy as np
import osr
import xarray as xr
import geopandas as gpd
from datetime import datetime
from datetime import timedelta  
import json
import dask
import itertools


# # Import CSO gdf (metadata) and df (daily SWE data) 

# In[3]:


gdf = gpd.read_file('../CSO_SNOTEL_sites.geojson')
df = pd.read_csv('../CSO_SNOTEL_data_SWEDmeters.csv') 
gdf.head()


# # Import baseline .par parameters 

# In[4]:


with open('par_base.json') as f:
    base = json.load(f)

base.keys()


# # Function to edit text files 
# ## Edit snowmodel.par and snowmodel.inc to run SnowModel as line -> original code

# In[5]:


#function to edit SnowModel Files other than .par
def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


# # Functions to adjust calibraiton parameters
# ## Edit snowmodel.par to run SnowModel as line -> Dave's code

# In[6]:


parFile = '/nfs/attic/dfh/Aragon2/mar2020_snowmodel-dfhill2/snowmodel.par'
incFile = '/nfs/attic/dfh/Aragon2/mar2020_snowmodel-dfhill2/code/snowmodel.inc'
compileFile = '/nfs/attic/dfh/Aragon2/mar2020_snowmodel-dfhill2/code/compile_snowmodel.script'
ctlFile = '/nfs/attic/dfh/Aragon2/mar2020_snowmodel-dfhill2/ctl_files/wo_assim/swed.ctl'
sweFile = '/nfs/attic/dfh/Aragon2/mar2020_snowmodel-dfhill2/outputs/wo_assim/swed.gdat'


# In[7]:


#Edit the par file to set parameters with new values
def edit_par(par_dict,parameter,new_value):
    lines = open(parFile, 'r').readlines()
    if par_dict[parameter][2] == 14 or par_dict[parameter][2] == 17     or par_dict[parameter][2] == 18 or par_dict[parameter][2] == 19     or par_dict[parameter][2] == 93 or par_dict[parameter][2] == 95     or par_dict[parameter][2] == 97 or par_dict[parameter][2] == 100     or par_dict[parameter][2] == 102 or par_dict[parameter][2] == 104     or par_dict[parameter][2] == 107 or par_dict[parameter][2] == 108:
        text = str(new_value)+'\n'
    else:
        text = str(new_value)+'\t\t\t!'+par_dict[parameter][1]
    lines[par_dict[parameter][2]] = text
    out = open(parFile, 'w')
    out.writelines(lines)
    out.close()


# In[8]:


#edit snowmodel.par
edit_par(base,'nx',np.shape(gdf)[0])
edit_par(base,'ny',1)
edit_par(base,'xmn',487200)
edit_par(base,'ymn',4690100)
edit_par(base,'dt',21600)
edit_par(base,'iyear_init',2014)
edit_par(base,'imonth_init',10)
edit_par(base,'iday_init',1)
edit_par(base,'xhour_init',0)
edit_par(base,'max_iter',7300)
edit_par(base,'met_input_fname','met/mm_wy_2014-2019.dat')
edit_par(base,'ascii_topoveg',1)
edit_par(base,'topo_ascii_fname','topo_vege/DEM_WY.asc')
edit_par(base,'veg_ascii_fname','topo_vege/NLCD2016_WY.asc')
edit_par(base,'xlat',40.2)
edit_par(base,'run_snowtran',0)
edit_par(base,'barnes_lg_domain',1)
edit_par(base,'snowmodel_line_flag',1)
edit_par(base,'lapse_rate','.28,1.2,2.8,4.2,4.5,4.4,4.0,3.8,3.7,3.4,2.6,0.87')#
edit_par(base,'prec_lapse_rate','0.4,0.4,0.46,0.41,0.27,0.24,0.21,0.17,0.22,0.32,0.43,0.39')#
edit_par(base,'print_inc',4)
edit_par(base,'print_var_01','y')
edit_par(base,'print_var_09','y')
edit_par(base,'print_var_10','y')
edit_par(base,'print_var_11','y')
edit_par(base,'print_var_12','y')
edit_par(base,'print_var_14','y')
edit_par(base,'print_var_18','y')
edit_par(base,'snowfall_frac',3)

##edit snowmodel.inc
#replace_line(incFile, 12, '      parameter (nx_max='+str(np.shape(gdf)[0]+1)+',ny_max=2)\n')
#replace_line(incFile, 12, '      parameter (nx_max=1383,ny_max=2477)\n')#full domain


##edit compile_snowmodel.script
#replace_line(compileFile, 16, '#pgf77 -O3 -mcmodel=medium -I$path -o ../snowmodel $path$filename1 $path$filename2 $path$filename3 $path$filename4 $path$filename5 $path$filename6 $path$filename7 $path$filename8 $path$filename9 $path$filename10\n')
#replace_line(compileFile, 20, 'gfortran -O3 -mcmodel=medium -I$path -o ../snowmodel $path$filename1 $path$filename2 $path$filename3 $path$filename4 $path$filename5 $path$filename6 $path$filename7 $path$filename8 $path$filename9 $path$filename10\n')


# # Function to compile/run SnowModel and extract relevant forcing parameters

# In[9]:


#Compile SnowModel - with Dave's code - should only have to do this once
get_ipython().run_line_magic('cd', '/nfs/attic/dfh/Aragon2/mar2020_snowmodel-dfhill2/code/')
#run compile script 
get_ipython().system(' ./compile_snowmodel.script')
get_ipython().run_line_magic('cd', '/nfs/attic/dfh/Aragon2/Notebooks/calibration_python')


# In[10]:


get_ipython().run_cell_magic('time', '', '\ndef runSnowModel():\n    %cd /nfs/attic/dfh/Aragon2/mar2020_snowmodel-dfhill2/\n    ! ./snowmodel\n    %cd  /nfs/attic/dfh/Aragon2/Notebooks/calibration_python\n\nrunSnowModel()')


# In[11]:


def get_mod_dims():
    #get model data from .ctl file 
    f=open(ctlFile)
    lines=f.readlines()
    nx = int(lines[9].split()[1])
    xll = int(float(lines[9].split()[3]))
    clsz = int(float(lines[9].split()[4]))
    ny = int(lines[10].split()[1])
    yll = int(float(lines[10].split()[3]))
    num_sim_days = int(lines[14].split()[1])
    st = datetime.strptime(lines[14].split()[3][3:], '%d%b%Y').date()
    ed = st + timedelta(days=(num_sim_days-1))
    print('nx=',nx,'ny=',ny,'xll=',xll,'yll=',yll,'clsz=',clsz,'num_sim_days=',num_sim_days,'start',st,'end',ed)
    f.close()
    return nx, ny, xll, yll, clsz, num_sim_days, st, ed

nx, ny, xll, yll, clsz, num_sim_days, st, ed = get_mod_dims()


# # Function to convert SnowModel output to numpy array
# 
# This function is to be used when running SnowModel as a line

# In[12]:


## Build a function to convert the binary model output to numpy array

def get_mod_output_line(inFile,stn):
    #open the grads model output file, 'rb' indicates reading from binary file
    grads_data = open(inFile,'rb')
    # convert to a numpy array 
    numpy_data = np.fromfile(grads_data,dtype='float32',count=-1)
    #close grads file 
    grads_data.close()
    #reshape the data 
    numpy_data = np.reshape(numpy_data,(num_sim_days,ny,nx))
    #swe only at station point
    data = np.squeeze(numpy_data[:,0,stn]) 

    return data


# # Function for calculating performance statistics

# In[13]:


#compute model performance metrics
def calc_metrics():
    swe_stats = np.zeros((5,np.shape(gdf)[0]))
    
    for i in range(np.shape(gdf)[0]):
        mod_swe = get_mod_output_line(sweFile,i)
        loc = gdf['code'][i]
        stn_swe = df[loc].values
        
        #remove days with zero SWE at BOTH the station and the SM pixel
        idx = np.where((stn_swe != 0) | (mod_swe != 0))
        mod_swe = mod_swe[idx]
        stn_swe = stn_swe[idx]
        
        #remove days where station has nan values 
        idx = np.where(~np.isnan(stn_swe))
        mod_swe = mod_swe[idx]
        stn_swe = stn_swe[idx]
        
        #R-squared value
        swe_stats[0,i] = np.corrcoef(stn_swe, mod_swe)[0,1]**2

        #mean bias error
        swe_stats[1,i] = (sum(mod_swe - stn_swe))/mod_swe.shape[0]

        #root mean squared error
        swe_stats[2,i] = np.sqrt((sum((mod_swe - stn_swe)**2))/mod_swe.shape[0])

        # Nash-Sutcliffe model efficiency coefficient, 1 = perfect, assumes normal data 
        nse_top = sum((mod_swe - stn_swe)**2)
        nse_bot = sum((stn_swe - mean(stn_swe))**2)
        swe_stats[3,i] = (1-(nse_top/nse_bot))

        # Kling-Gupta Efficiency, 1 = perfect
        kge_std = (np.std(mod_swe)/np.std(stn_swe))
        kge_mean = (mean(mod_swe)/mean(stn_swe))
        kge_r = corrcoef(stn_swe,mod_swe)[1,0]
        swe_stats[4,i] = (1 - (sqrt((kge_r-1)**2)+((kge_std-1)**2)+(kge_mean-1)**2))
        
    return swe_stats

swe_stats = calc_metrics()


# # Create dataframe of calibration parameters and run calibration

# In[16]:


#Calibration parameters

#snowfall_frac = [1,2,3]
#if = 1 -> 
#T_threshold = arange(float(base ['T_threshold'][0])-2,float(base ['T_threshold'][0])+2,1)
#if = 3 -> base['T_Left,T_Right']
#figure out how to parse these 

#snowfall_frac = [3]
#T_L_R = [base['T_Left,T_Right'][0],'-2,1','-2,2','-2,3','-1,1','-1,2','-1,3','0,2','0,3']


#wind_lapse_rate = arange(float(base ['wind_lapse_rate'][0]),
#                  float(base ['wind_lapse_rate'][0])+1.5,.5)
gap_frac = arange(0,.8,.2)

lat_solar_flag = [0,1]

#cannot use lonwave or shortwave obs with barnes large domain flag needed to run line mode of SM
# use_shortwave_obs = [0,1]

# use_longwave_obs = [0,1]

lapse_rate= [base['lapse_rate'][0],
             '.28,1.2,2.8,4.2,4.5,4.4,4.0,3.8,3.7,3.4,2.6,0.87']
prec_lapse_rate = [base ['prec_lapse_rate'][0],
                   '0.4,0.4,0.46,0.41,0.27,0.24,0.21,0.17,0.22,0.32,0.43,0.39']

ro_snowmax=arange(float(base ['ro_snowmax'][0])-200,
                  float(base ['ro_snowmax'][0])+200,25)

cf_precip_scalar=arange(float(base ['cf_precip_scalar'][0])-.3,
                        float(base ['cf_precip_scalar'][0])+.4,.1)
# add cf_precip_flag = 3
ro_adjust=arange(float(base ['ro_adjust'][0])-1,
                 float(base ['ro_adjust'][0])+2,1)

Total_runs = len(lat_solar_flag) * len(gap_frac) * len(lapse_rate) * len(prec_lapse_rate) * len(ro_snowmax)*len(cf_precip_scalar)*len(ro_adjust)
print('Total number of calibration runs = ',Total_runs)
print('This will take approximately',Total_runs*14/60/60,'hours')
gap_frac


# In[33]:


parameters = [lat_solar_flag,lapse_rate,
              prec_lapse_rate,ro_snowmax,cf_precip_scalar,ro_adjust,gap_frac]
data = list(itertools.product(*parameters))
input_params = pd.DataFrame(data,columns=['lat_solar_flag','lapse_rate','prec_lapse_rate',
                                          'ro_snowmax','cf_precip_scalar','ro_adjust','gap_frac'])


# In[15]:


timestamp = str(datetime.date(datetime.now()))+'_full_set'

#save input parameters as csv
input_params.to_csv('cal_params_'+timestamp+'.csv',index=False)


# In[36]:


get_ipython().run_cell_magic('time', '', "%cd /nfs/attic/dfh/Aragon2/mar2020_snowmodel-dfhill2/\n\nswe_stats = np.empty([shape(input_params)[0],5,np.shape(gdf)[0]])\nfor i in range(np.shape(input_params)[0]):\n    print(i)\n    edit_par(base,'lat_solar_flag',input_params.lat_solar_flag[i])\n#     edit_par(base,'use_shortwave_obs',input_params.use_shortwave_obs[i])\n#     edit_par(base,'use_longwave_obs',input_params.use_longwave_obs[i])\n    edit_par(base,'lapse_rate',input_params.lapse_rate[i])\n    edit_par(base,'prec_lapse_rate',input_params.prec_lapse_rate[i])\n    edit_par(base,'ro_snowmax',input_params.ro_snowmax[i])\n    edit_par(base,'cf_precip_scalar',input_params.cf_precip_scalar[i])\n    edit_par(base,'ro_adjust',input_params.ro_adjust[i])\n    edit_par(base,'gap_frac',input_params.gap_frac[i])\n    ! nohup ./snowmodel\n    swe_stats[i,:,:] = calc_metrics()\n    print(swe_stats[i,:,:])\n\n%cd  /nfs/attic/dfh/Aragon2/Notebooks/calibration_python")


# # Save output as netcdf

# In[17]:


#Turn NDarray into xarray 
calibration_run = np.arange(0,swe_stats.shape[0],1)
metric = ['MAE','MBE','RMSE','NSE','KGE']
station = gdf['code'].values

cailbration = xr.DataArray(
    swe_stats,
    dims=('calibration_run', 'metric', 'station'), 
    coords={'calibration_run': calibration_run, 
            'metric': metric, 'station': station})

cailbration.attrs['long_name']= 'Calibration performance metrics'
cailbration.attrs['standard_name']= 'cal_metrics'

d = OrderedDict()
d['calibration_run'] = ('calibration_run', calibration_run)
d['metric'] = ('metric', metric)
d['station'] = ('station', station)
d['cal_metrics'] = cailbration

ds = xr.Dataset(d)
ds.attrs['description'] = "SnowModel line calibration performance metrics"
ds.attrs['calibration_parameters'] = "ro_snowmax,cf_precip_scalar,ro_adjust"
ds.attrs['model_parameter'] = "SWE [m]"

ds.calibration_run.attrs['standard_name'] = "calibration_run"
ds.calibration_run.attrs['axis'] = "N"

ds.metric.attrs['long_name'] = "calibration_metric"
ds.metric.attrs['axis'] = "metric"

ds.station.attrs['long_name'] = "station_id"
ds.station.attrs['axis'] = "station"

ds.to_netcdf('calibration_'+timestamp+'.nc', format='NETCDF4', engine='netcdf4')


# In[ ]:




