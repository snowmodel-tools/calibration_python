#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all of the python packages used in this workflow.
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
import itertools
import requests
from sklearn.metrics import r2_score
import time


# In[2]:


########## USER ###########
# Select modeling domain ('WY', 'UT', 'OR', 'WA')
domain = 'SUS'

# SM location 
SMpath = '/nfs/attic/dfh/Aragon2/CSOsm/jan2021_snowmodel-dfhill_'+domain+'/'

metFname = 'mm_'+domain+'_2016-2019.dat'


# In[3]:


dataPath = '/nfs/attic/dfh/Aragon2/CSOdmn/'+domain+'/'
# Outfile path
outpath = '/nfs/attic/dfh/Aragon2/CSOcal/'+domain+'/'

#path to CSO domains
domains_resp = requests.get("https://raw.githubusercontent.com/snowmodel-tools/preprocess_python/master/CSO_domains.json")
domains = domains_resp.json()

#start date    
st_dt = domains[domain]['st']
#end date
ed_dt = domains[domain]['ed']

#Snotel bounding box
Bbox = domains[domain]['Bbox']

# Snotel projection
stn_proj = domains[domain]['stn_proj']
# model projection
mod_proj = domains[domain]['mod_proj']


# # Import CSO gdf (metadata) and df (daily SWE data) 

# In[4]:


gdf = gpd.read_file(dataPath+ 'CSO_SNOTEL_sites_'+domain+'.geojson')
df = pd.read_csv(dataPath+'SNOTEL_data_SWEDmeters'+st_dt+'_'+ed_dt+'.csv') 
gdf.head()


# # Function to edit text files 

# In[6]:


#Edit the par file to set parameters with new values
def edit_par(par_dict,parameter,new_value,parFile):
    lines = open(parFile, 'r').readlines()
    if par_dict[parameter][2] == 14 or par_dict[parameter][2] == 17     or par_dict[parameter][2] == 18 or par_dict[parameter][2] == 19     or par_dict[parameter][2] == 93 or par_dict[parameter][2] == 95     or par_dict[parameter][2] == 97 or par_dict[parameter][2] == 100     or par_dict[parameter][2] == 102 or par_dict[parameter][2] == 104     or par_dict[parameter][2] == 107 or par_dict[parameter][2] == 108     or par_dict[parameter][2] == 147 or par_dict[parameter][2] == 148     or par_dict[parameter][2] == 149:
        text = str(new_value)+'\n'
    else:
        text = str(new_value)+'\t\t\t!'+par_dict[parameter][1]
    lines[par_dict[parameter][2]] = text
    out = open(parFile, 'w')
    out.writelines(lines)
    out.close()


# In[8]:


#function to edit SnowModel Files other than .par
def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


# # Import baseline .par parameters 

# In[9]:


parFile = SMpath+'snowmodel.par'
incFile = SMpath+'code/snowmodel.inc'
compileFile = SMpath+'code/compile_snowmodel.script'
sweFile = SMpath+'outputs/wo_assim/swed.gdat'
micrometFile = SMpath+'code/micromet_code.f'
ctlFile = SMpath+'ctl_files/wo_assim/swed.ctl'
codepath = SMpath+'code'
calpath = '/nfs/attic/dfh/Aragon2/Notebooks/calibration_python'


# In[10]:


#path to base par
basepar_resp = requests.get("https://raw.githubusercontent.com/snowmodel-tools/calibration_python/master/par_base.json")
base = basepar_resp.json()
#set par to baseline 
for key in base:
    edit_par(base,key,base[key][0],parFile)
base.keys()


# # Adjust calibraiton parameters
# ## Edit snowmodel.par to run SnowModel as line

# In[11]:


#edit snowmodel.par to run SM as a line 
# spatial inputs
edit_par(base,'nx',np.shape(gdf)[0],parFile)
edit_par(base,'ny',1,parFile)
edit_par(base,'xmn',domains[domain]['xll'],parFile)
edit_par(base,'ymn',domains[domain]['yll'],parFile)
edit_par(base,'deltax',domains[domain]['cellsize'],parFile)
edit_par(base,'deltay',domains[domain]['cellsize'],parFile)
# temporal inputs
edit_par(base,'dt',21600,parFile) #seconds per model time step
edit_par(base,'iyear_init',datetime.strptime(st_dt,'%Y-%m-%d').year,parFile)
edit_par(base,'imonth_init',datetime.strptime(st_dt,'%Y-%m-%d').month,parFile)
edit_par(base,'iday_init',datetime.strptime(st_dt,'%Y-%m-%d').day,parFile)
edit_par(base,'xhour_init',datetime.strptime(st_dt,'%Y-%m-%d').hour,parFile)
edit_par(base,'max_iter',(datetime.strptime(ed_dt,'%Y-%m-%d')-
    datetime.strptime(st_dt,'%Y-%m-%d')).days*4+4,parFile)
# paths 
edit_par(base,'met_input_fname','../../CSOdmn/'+domain+'/'+metFname,parFile)
edit_par(base,'ascii_topoveg',1,parFile)
edit_par(base,'topo_ascii_fname','../../CSOdmn/'+domain+'/DEM_'+domain+'_line.asc',parFile)
edit_par(base,'veg_ascii_fname','../../CSOdmn/'+domain+'/NLCD2016_'+domain+'_line.asc',parFile)
edit_par(base,'lat_file_path','../../CSOdmn/'+domain+'/grid_lat_'+domain+'_line.asc',parFile)
edit_par(base,'lon_file_path','../../CSOdmn/'+domain+'/grid_lon_'+domain+'_line.asc',parFile)
edit_par(base,'snowmodel_line_file','../../CSOdmn/'+domain+'/snowmodel_line_pts.dat',parFile)
#other flags to calibrate SM as line
edit_par(base,'xlat',round(domains[domain]['Bbox']['latmin']+(domains[domain]['Bbox']['latmax']-domains[domain]['Bbox']['latmin'])/2,2),parFile)
edit_par(base,'run_snowtran',0,parFile)
edit_par(base,'barnes_lg_domain',1,parFile)
edit_par(base,'lat_solar_flag',1,parFile)
edit_par(base,'snowmodel_line_flag',1,parFile)
edit_par(base,'print_inc',4,parFile)
edit_par(base,'print_var_01','n',parFile)#tair
edit_par(base,'print_var_09','n',parFile)#prec
edit_par(base,'print_var_10','n',parFile)#rain
edit_par(base,'print_var_11','n',parFile)#sprec
edit_par(base,'print_var_12','n',parFile)#swemelt
edit_par(base,'print_var_14','n',parFile)#runoff
edit_par(base,'print_var_18','y',parFile)#swed
edit_par(base,'cf_precip_flag',3,parFile)
edit_par(base,'UTC_flag',1,parFile)

##edit snowmodel.inc
replace_line(incFile, 12, '      parameter (nx_max='+str(np.shape(gdf)[0]+1)+',ny_max=2)\n')
replace_line(incFile, 41, '      parameter (nz_max=10)\n') 
#replace_line(incFile, 12, '      parameter (nx_max=1383,ny_max=2477)\n')#full domain


##edit compile_snowmodel.script
#replace_line(compileFile, 16, '#pgf77 -O3 -mcmodel=medium -I$path -o ../snowmodel $path$filename1 $path$filename2 $path$filename3 $path$filename4 $path$filename5 $path$filename6 $path$filename7 $path$filename8 $path$filename9 $path$filename10\n')
#replace_line(compileFile, 20, 'gfortran -O3 -mcmodel=medium -I$path -o ../snowmodel $path$filename1 $path$filename2 $path$filename3 $path$filename4 $path$filename5 $path$filename6 $path$filename7 $path$filename8 $path$filename9 $path$filename10\n')


# # Compile SnowModel

# In[12]:


#function to calibrate and run SM
st_time = time.time()
def runcalSnowModel():
    get_ipython().run_line_magic('cd', '$codepath')
    #run compile script 
    get_ipython().system(' ./compile_snowmodel.script')
    get_ipython().run_line_magic('cd', '$SMpath')
    get_ipython().system(' ./snowmodel')
runcalSnowModel()
get_ipython().run_line_magic('cd', '$calpath')
total_time = time.time()-st_time
print(total_time)


# # Test Run SnowModel (optional)

# In[13]:


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

# # Edit
# Compare this to the assim code

# In[17]:


## Build a function to convert the binary model output to numpy array

def get_mod_output_lines(inFile):
    #open the grads model output file, 'rb' indicates reading from binary file
    grads_data = open(inFile,'rb')
    # convert to a numpy array 
    numpy_data = np.fromfile(grads_data,dtype='float32',count=-1)
    #close grads file 
    grads_data.close()
    #reshape the data
    numpy_data = np.reshape(numpy_data,(num_sim_days,1,np.shape(gdf)[0]))
    #swe only at station point
    data = np.squeeze(numpy_data[:,0,:]) 

    return data



# # Function for calculating performance statistics
# 
# MAke sure I am using the correct r -> look at stats notes
# 

# In[ ]:


#compute model performance metrics
def calc_metrics():
    swe_stats = np.zeros((5,np.shape(gdf)[0]))
    
    for i in range(np.shape(gdf)[0]):
        mod_swes = get_mod_output_lines(sweFile)
        if mod_swes.ndim == 1:
            mod_swe = mod_swes
        else:
            mod_swe = mod_swes[:,i]
        loc = gdf['code'][i]
        stn_swe = df[loc].values
        
        #remove days with zero SWE at BOTH the station and the SM pixel
        idx = np.where((stn_swe != 0) & (mod_swe != 0))
        mod_swe = mod_swe[idx]
        stn_swe = stn_swe[idx]
        
        #remove days where station has nan values 
        idx = np.where(~np.isnan(stn_swe))
        mod_swe = mod_swe[idx]
        stn_swe = stn_swe[idx]
        
        #R-squared value
        swe_stats[0,i] = sm.OLS(mod_swe, stn_swe).fit().rsquared

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


# # Create dataframe of calibration parameters
# 
# Add the vegetation parameters
# 

# In[25]:


#Calibration parameters

#snowfall_frac = [1,2,3]
#if = 1 -> 
#T_threshold = arange(float(base ['T_threshold'][0])-2,float(base ['T_threshold'][0])+2,1)
#if = 3 -> base['T_Left,T_Right']
#figure out how to parse these 

#snowfall_frac = [3]
#T_L_R = [base['T_Left,T_Right'][0],'-2,1','-2,2','-2,3','-1,1','-1,2','-1,3','0,2','0,3']

################# MultiLayer Snowpack ########################
max_layers = [1,10]
#print(max_layers, 'Max Layers')
#print(len(max_layers),'# of Values')

######################### Wind ###############################
# wind_lapse_rate = arange(float(base ['wind_lapse_rate'][0]),
#                  float(base ['wind_lapse_rate'][0])+2.5,.5)

######################### Solar ###############################

#lat_solar_flag = [0,1]

#################### Temperature ############################
#lapse_rate= [base['lapse_rate'][0],tlapse]

####################### Snow Density #######################
ro_snowmax=arange(float(base ['ro_snowmax'][0])-150,
                  float(base ['ro_snowmax'][0])+150,50)
ro_adjust=arange(float(base ['ro_adjust'][0])-1,
                 float(base ['ro_adjust'][0])+2,1)

#################### Precipitation ############################
cf_precip_scalar=arange(float(base ['cf_precip_scalar'][0])-.4,
                        float(base ['cf_precip_scalar'][0])+.4,.1)
# add cf_precip_flag = 3
#prec_lapse_rate = [base ['prec_lapse_rate'][0],plapse]

#################### Vegetation ############################
conifer_lai = arange(1.5,3,0.3)

gap_frac = arange(0.2,1,.2)

Total_runs = len(max_layers) * len(ro_snowmax) * len(gap_frac) * len(ro_adjust)*len(cf_precip_scalar)*len(conifer_lai)
print('Total number of calibration runs = ',Total_runs)

# in the test model run -> get the wall time to estimate how long the calibration will take
wall_time = total_time
print('This will take approximately',Total_runs*wall_time/60/60,'hours')


# In[27]:


# Parameters used in this calibration run
parameters = [max_layers,ro_snowmax,ro_adjust,cf_precip_scalar,conifer_lai,gap_frac]
data = list(itertools.product(*parameters))
input_params = pd.DataFrame(data,columns=                            ['max_layers','ro_snowmax','ro_adjust',                             'cf_precip_scalar','conifer_lai','gap_frac'])


# In[ ]:


timestamp = str(datetime.date(datetime.now()))

#save input parameters as csv
input_params.to_csv(outpath+'cal_params_'+timestamp+'.csv',index=False)


# # Run calibration

# In[ ]:


get_ipython().run_cell_magic('time', '', "%cd $SMpath\n\ntop_swe = np.empty([len(input_params),num_sim_days,np.shape(gdf)[0]])        \nswe_stats = np.empty([shape(input_params)[0],5,np.shape(gdf)[0]])\n\nif np.shape(gdf)[0] == 1:\n    for i in range(np.shape(input_params)[0]):\n        print(i+1, 'of', Total_runs)\n        edit_par(base,'max_layers',input_params.max_layers[i],parFile)\n        edit_par(base,'lapse_rate',input_params.lapse_rate[i],parFile)\n        edit_par(base,'prec_lapse_rate',input_params.prec_lapse_rate[i],parFile)\n        edit_par(base,'ro_snowmax',input_params.ro_snowmax[i],parFile)\n        edit_par(base,'cf_precip_scalar',input_params.cf_precip_scalar[i],parFile)\n        edit_par(base,'ro_adjust',input_params.ro_adjust[i],parFile)\n        edit_par(base,'gap_frac',input_params.gap_frac[i],parFile)\n        replace_line(micrometFile, 3336,'      data vlai_summer /'+str(input_params.conifer_lai[i])+', 2.5, 2.5, 1.5, 1.0/\\n')\n        replace_line(micrometFile, 3337,'      data vlai_winter /'+str(input_params.conifer_lai[i])+', 0.5, 1.5, 1.5, 1.0/\\n')\n        %cd $codepath\n        #run compile script \n        ! ./compile_snowmodel.script\n        %cd $SMpath\n        ! nohup ./snowmodel\n        swe_stats[i,:,:] = calc_metrics()\n        mod_swe = get_mod_output_lines(sweFile)\n        top_swe[i,:,:] = mod_swe.reshape((len(mod_swe), 1))\nelse:\n    for i in range(np.shape(input_params)[0]):\n        print(i+1, 'of', Total_runs)\n        edit_par(base,'max_layers',input_params.max_layers[i],parFile)\n        edit_par(base,'lapse_rate',input_params.lapse_rate[i],parFile)\n        edit_par(base,'prec_lapse_rate',input_params.prec_lapse_rate[i],parFile)\n        edit_par(base,'ro_snowmax',input_params.ro_snowmax[i],parFile)\n        edit_par(base,'cf_precip_scalar',input_params.cf_precip_scalar[i],parFile)\n        edit_par(base,'ro_adjust',input_params.ro_adjust[i],parFile)\n        edit_par(base,'gap_frac',input_params.gap_frac[i],parFile)\n        replace_line(micrometFile, 3336,'      data vlai_summer /'+str(input_params.conifer_lai[i])+', 2.5, 2.5, 1.5, 1.0/\\n')\n        replace_line(micrometFile, 3337,'      data vlai_winter /'+str(input_params.conifer_lai[i])+', 0.5, 1.5, 1.5, 1.0/\\n')\n        %cd $codepath\n        #run compile script \n        ! ./compile_snowmodel.script\n        %cd $SMpath\n        ! nohup ./snowmodel\n        swe_stats[i,:] = np.squeeze(calc_metrics())\n        mod_swe = get_mod_output_lines(sweFile)\n        top_swe[i,:] = mod_swe\n%cd $calpath")


# # Save output as netcdf

# In[ ]:


#Turn NDarray into xarray 
calibration_run = np.arange(0,swe_stats.shape[0],1)
metric = ['R2','MBE','RMSE','NSE','KGE']
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

ds.to_netcdf(outpath+'calibration_'+timestamp+'.nc', format='NETCDF4', engine='netcdf4')


# In[ ]:


calibration_run = np.arange(0,top_swe.shape[0],1)
sim_day = np.arange(0,top_swe.shape[1],1)
station = gdf['code'].values

swe = xr.DataArray(
    top_swe,
    dims=('calibration_run', 'sim_day', 'station'), 
    coords={'calibration_run': calibration_run, 
            'sim_day': sim_day, 'station': station})

swe.attrs['long_name']= 'swe timeseries [m]'
swe.attrs['standard_name']= 'swe'

d = OrderedDict()
d['calibration_run'] = ('calibration_run', calibration_run)
d['sim_day'] = ('sim_day', sim_day)
d['station'] = ('station', station)
d['cal_metrics'] = swe

ds = xr.Dataset(d)
ds.attrs['description'] = "SnowModel swe"
ds.attrs['model_parameter'] = "SWE [m]"

ds.calibration_run.attrs['standard_name'] = "calibration_run"
ds.calibration_run.attrs['axis'] = "N"

ds.sim_day.attrs['long_name'] = "sumilation_day"
ds.sim_day.attrs['axis'] = "sim_day"

ds.station.attrs['long_name'] = "station_id"
ds.station.attrs['axis'] = "station"

ds.to_netcdf(outpath+'swe_'+timestamp+'.nc', format='NETCDF4', engine='netcdf4')


# In[ ]:




