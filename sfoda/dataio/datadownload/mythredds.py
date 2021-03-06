"""
Various tools for interacting with opendap thredds datasets

Matt Rayson
Stanford University
October 2014
"""

import os
import time
import collections
import numpy as np
from netCDF4 import Dataset, MFDataset, date2num, date2index
from cftime import num2pydate
from datetime import datetime
from sfoda.utils import othertime

import pdb

# Parallel modules
#import gevent
#from gevent.pool import Pool
#from gevent import monkey
#
#monkey.patch_all()




####
# Utility functions
####
def hasdim(nc,dim):
    return dim in nc.dimensions
def hasvar(nc,var):
    return var in nc.variables
def gettime(nc,timename):
    t = nc.variables[timename]
    return num2pydate(t[:],t.units)



class GridDAP(object):
    """
    General class for extracting gridded opendap data
    """

    timecoord = 'time'
    xcoord = 'lon'
    ycoord = 'lat'
    zcoord = None

    gridfile = None
    multifile = False

    # Set this to pass your own multi-file object
    MF = None

    # Try and grab all steps together
    graball = True

    def __init__(self,url,**kwargs):

        self.__dict__.update(kwargs)

        # Open the opendap dataset
        if isinstance(url,str) or isinstance(url, str):
            self._nc = Dataset(url)
        elif isinstance(url,Dataset):
            # we're passing Dataset object
            self._nc = url
        elif isinstance(url,list) and self.multifile:
            # We are passing it a list for multifile variables
            self._nc = Dataset(url[0])
            self._ncfiles = url # List of file names

        else:   
            raise Exception('Unknown input url type')

        # Open the grid netcdf (if it exists)
        if not self.gridfile==None:
            self._gridnc = Dataset(self.gridfile)
        else:
            self._gridnc = self._nc

        self.get_coords()


    def get_coords(self):
        """ Download the coordinates"""

        #print self._gridnc.variables.keys()
        self.X = self._gridnc.variables[self.xcoord][:]
        self.Y = self._gridnc.variables[self.ycoord][:]

        if not self.zcoord == None:
            self.Z = self._gridnc.variables[self.zcoord][:]

    def get_time_indices(self,trange):
        """
        Find the time indices
        """

        if not self.multifile:
            self.time = gettime(self._nc,self.timecoord)

        else:
            if self.MF==None:
                self.MF = MFncdap(self._ncfiles,timevar=self.timecoord)

            self.time = self.MF.time

        # Time
        if trange==None:
            self.t1=0
            self.t2 = self.time.shape[0]
        else:
            self.t1 = othertime.findNearest(trange[0],self.time)
            self.t2 = othertime.findNearest(trange[1],self.time)

        if self.t1==self.t2:
            self.t2+=1

        self.nt = self.t2 - self.t1

        print(trange)

    def get_indices(self,xrange,yrange,zrange):
        """ Find the domain indices"""

        # Check if X/Y are 1D or gridded

        if self.X.ndim==1:
            if xrange is None:
                self.x1=0
                self.x2=self.X.shape[0]
            else:
                self.x1 = self.find_nearest_1d(self.X,xrange[0])
                self.x2 = self.find_nearest_1d(self.X,xrange[1])

            if yrange is None:
                self.y1=0
                self.y2 = self.Y.shape[0]
            else:
                y1 = self.find_nearest_1d(self.Y,yrange[0])
                y2 = self.find_nearest_1d(self.Y,yrange[1])
                self.y1 = min([y1,y2])
                self.y2 = max([y1,y2])
        
        elif self.X.ndim==2:
            if xrange == None or yrange == None:
                self.x1 = 0
                self.x2 = self.X.shape[1]
                self.y1 = 0
                self.y2 = self.Y.shape[0]
            else:
                ind = self.find_nearest_2d(\
                    self.X,self.Y,[xrange[0],yrange[0]])
                self.y1,self.x1 = ind[0][0],ind[0][1]
                ind = self.find_nearest_2d(\
                    self.X,self.Y,[xrange[1],yrange[1]])
                self.y2,self.x2 = ind[0][0],ind[0][1]

        self.nx = self.x2-self.x1
        self.ny = self.y2-self.y1

        if not self.zcoord == None:
            if zrange==None:
                self.z1=0
                self.z2=self.Z.shape[0]
            else:
                self.z1 = self.find_nearest_1d(zrange[0],self.Z)
                self.z2 = self.find_nearest_1d(zrange[1],self.Z)
                if self.z1==self.z2:
                    self.z2+=1
            self.nz = self.z2 - self.z1


    def get_data(self,varname,xrange,yrange,zrange,trange, rawvar=None):
        """
        Download the actual data

        Set xrange/yrange/zrange/trange=None to get all
        
        """
        
        self.get_indices(xrange,yrange,zrange)
        self.get_time_indices(trange)

        # Store the local coordinates
        self.localtime = self.time[self.t1:self.t2]

        if not self.multifile:
            return self.get_data_singlefile(varname,self._nc,self.t1,self.t2)
        else:
            return self.get_data_multifile(varname, rawvar=rawvar) 

    def get_data_multifile(self,varname, rawvar=None):
        """
        Download the data from multiple files and stick in one array

        """
        if self.zcoord==None:
            sz = (self.nt,self.ny,self.nx)
        else:
            sz = (self.nt,self.nz,self.ny,self.nx)

        data = np.zeros(sz)

        tindex,ncurl,tslice_dict = self.MF(self.localtime.tolist(), var=rawvar)


        ####
        ## Download time chunks of data in parallel
        ###

        ## Build a list of tuples with the record indices
        #recs = []
        #p1 = 0
        #for ff in tslice_dict.keys():
        #    t1,t2 = tslice_dict[ff]
        #    p2 = p1+t2-t1
        #    #print '\t Downloading from file:\n%s'%ff
        #    #data[p1:p2+1,...] = self.get_data_singlefile(varname,nc,t1,t2+1)

        #    p1=p2+1
        #    _nc = Dataset(ff)
        #    recs.append((ff, p1,p2,t1,t2, _nc))


        #def get_data_parallel(rec):
        #    ''' Parallel wrapper function'''
        #    ff, p1, p2, t1, t2, nc = rec

        #    #nc = Dataset(ff)
        #    print '\t Downloading from file (parallel):\n%s'%ff
        #    data_tmp = self.get_data_singlefile(varname,nc,t1,t2+1)
        #    #nc.close()

        #    return data_tmp

        ## Do the work on many threads...
        #pool = Pool(32)
        #datas = pool.map(get_data_parallel, recs)

        ## Insert the list of data chunks back into the array
        #for rec, data_i in zip(recs, datas):
        #    ff, p1, p2, t1, t2, nc = rec
        #    data[p1:p2+1,...] = data_i
        #    nc.close()

        ###
        # Download time chunks of data
        ###
        def openfile(ff):
            try:
                nc = Dataset(ff)
            except:
                print('Ouch! Server says no... I''ll retry...')
                time.sleep(1.1)
                nc = openfile(ff)

            return nc

        p1 = 0
        for ff in list(tslice_dict.keys()):
            nc = openfile(ff)
            t1,t2 = tslice_dict[ff]
            p2 = p1+t2-t1
            print('\t Downloading from file:\n%s'%ff)
            data[p1:p2+1,...] = self.get_data_singlefile(varname,nc,t1,t2+1)

            p1=p2+1
            nc.close()
        
        ####
        ## Download data step-by-step (slow)
        ####
        #ncold = ncurl[0]
        #nc = Dataset(ncold)
        #ii=-1
        #for t,name in zip(tindex,ncurl):
        #    ii+=1
        #    if not ncold == name:
        #        nc.close()
        #        nc = Dataset(name)
        #    else:
        #        ncold = name

        #    print '\t Downloading time: ', t
        #    data[ii,...] = self.get_data_singlefile(varname,nc,t,t+1)

        #nc.close()
        return data
            
    def get_data_singlefile(self,varname,nc,t1,t2):  
        """
        Download the data from a single file
        """

        def get_2d(data):
            try:
                # Do step-by-step (slower)
                data = nc.variables[varname]\
                    [self.y1:self.y2,self.x1:self.x2]

            except:
                print('Ouch! Server says no... I''ll retry...')
                time.sleep(0.1)
                data = get_2d(data)

            return data


        def get_3d_step(data,tstart):
            try:
                # Do step-by-step (slower)
                print('Downloading step-by-step...')
                for ii,tt in enumerate(range(tstart,t2)):
                    print(tt, t2)
                    data[ii,...] = nc.variables[varname]\
                        [tt,self.y1:self.y2,self.x1:self.x2]

            except:
                print('Ouch! Server says no... I''ll retry...')
                time.sleep(0.1)
                data = get_3d_step(data,tt)

            return data


        def get_4d_step(data,tstart):
            try:
                # Do step-by-step (slower)
                print('Downloading step-by-step...')
                for ii,tt in enumerate(range(tstart,t2)):
                    print(tt, t2)
                    data[ii,...] = nc.variables[varname]\
                        [tt,self.z1:self.z2,self.y1:self.y2,self.x1:self.x2]

            except:
                print('Ouch! Server says no... I''ll retry...')
                time.sleep(0.1)
                data = get_4d_step(data,tt)

            return data

        ndim = nc.variables[varname].ndim
        if ndim==2:
            data = np.zeros((1, self.ny, self.nx))
            data = get_2d(data)

        elif ndim==3:
            try:
                if self.graball:
                    data = nc.variables[varname]\
                        [t1:t2,self.y1:self.y2,self.x1:self.x2]
                else:
                    data = np.zeros((self.nt, self.ny, self.nx))
                    data = get_3d_step(data,t1)
            except:
                data = np.zeros((self.nt, self.ny, self.nx))
                data = get_3d_step(data,t1)

        elif ndim==4:
            try:
                if self.graball:
                    data = nc.variables[varname]\
                        [t1:t2,self.z1:self.z2,\
                        self.y1:self.y2,self.x1:self.x2]
                else:
                    data = np.zeros((self.nt, self.nz, self.ny, self.nx))
                    data = get_4d_step(data,t1)
            except:
                data = np.zeros((self.nt, self.nz, self.ny, self.nx))
                data = get_4d_step(data,t1)

            #except:
            ## Do step-by-step (slower)
            #data = np.zeros((self.nt, self.nz, self.ny, self.nx))
            #print 'Downloading step-by-step...'
            #for ii,tt in enumerate(range(t1,t2)):
            #    print tt, t2
            #    data[ii,...] = nc.variables[varname]\
            #        [tt,self.z1:self.z2,self.y1:self.y2,self.x1:self.x2]

            self.localZ = self.Z[self.z1:self.z2]

        if self.X.ndim==1:
            self.localX = self.X[self.x1:self.x2]
            self.localY = self.Y[self.y1:self.y2]
        else:
            self.localX = self.X[self.y1:self.y2,self.x1:self.x2]
            self.localY = self.Y[self.y1:self.y2,self.x1:self.x2]

        return data

    def find_nearest_1d(self,data,value):
        dist = np.abs(data-value)
        return np.argwhere(dist==dist.min())[0][0]

    def find_nearest_2d(self,xdata,ydata,xy):
        dist = np.sqrt( (xy[0]-xdata)**2. + (xy[1]-ydata)**2.)
        return np.argwhere(dist==dist.min())


class GetDAP(object):
    """
    High level class for downloading data
    """
    ncurl = None # file location (can be url or local)
    type = 'ocean' # Ocean or atmoshpere
    multifile = False

    # Ocean variable names (name of variables in file)
    u = 'u'
    v = 'v'
    temp = 'temp'
    salt = 'salt'
    ssh = 'ssh'
    # Default variables to extract
    oceanvars = ['ssh','u','v','temp','salt']

    timedim = 'time'

    # Atmosphere variable names
    uwind = 'uwind'
    vwind = 'vwind'
    tair = 'tair'
    pair = 'pair'
    rh = 'rh'
    cloud = 'cloud'
    rain = 'rain'

    # Default variables to extract
    airvars = ['uwind','vwind','tair','pair','rh','cloud','rain']

    tformat = '%Y%m%d.%H%M%S'

    # Location of cached grid file
    gridfile = None

    # Set this to pass your own multi-file object
    MF = None

    def __init__(self,variables=None,**kwargs):
        """
        Initialise the variables to outfile
        """
        self.__dict__.update(kwargs)

        # Open the file
        if not self.multifile:
            try:
                self._nc = Dataset(self.ncurl)
            except:
                try:
                    self._nc = MFDataset(self.ncurl, aggdim=self.timedim)
                except:
                    self._nc = MFDataset(self.ncurl,)
        else:
            self._nc = Dataset(self.ncurl[0])
            self._ncfiles = self.ncurl

        # Set the variables that need downloading
        if variables==None and self.type=='ocean':
            self.variables = self.oceanvars
        elif variables==None and self.type=='atmosphere':
            self.variables = self.airvars
        else:
            self.variables = variables

        # For each variable:
        # Get the coordinate variables
        
        # Initiate the griddap class for each variable
        #   (this does most of the work)
        self.init_var_grids()

    def __call__(self,xrange,yrange,trange,zrange=None,outfile=None):    
        """
        Download the data, save if outfile is specified
        """

        trange=self.check_trange(trange)

        # Load the data for all variable and write
        for vv in self.variables:
            data = self.load_data(vv,xrange,yrange,zrange,trange)

            # Write the data
            if not outfile == None:
                self.write_var(outfile,vv,data)
        
        return data


    def load_data(self,varname,xr,yr,zr,tr):
        ncvarname = getattr(self,varname)
        dap = getattr(self,'_%s'%varname)
        print('Loading variable: (%s) %s...'%(varname,ncvarname))

        return dap.get_data(ncvarname,xr,yr,zr,tr, rawvar=varname)

    def init_var_grids(self):
        """
        Initialise each variable into an object with name "_name"
        """
        for vv in self.variables:
            print('loading grid data for variable: %s...'%getattr(self,vv))
            attrname = '_%s'%vv

            if self.multifile:
                nc = self.MF.get_filename_only(var=vv)
                print(nc)
                self._nc = Dataset(nc)
            else:
                nc = self._nc
 
            timecoord,xcoord,ycoord,zcoord = \
                self.get_coord_names(getattr(self,vv))
               
            dap = GridDAP(nc,xcoord=xcoord,ycoord=ycoord,\
                timecoord=timecoord,zcoord=zcoord,\
                gridfile=self.gridfile,multifile=self.multifile,\
                MF=self.MF)

            setattr(self,attrname,dap)
 

    def check_trange(self,trange):
        if isinstance(trange[0],str):
            t1 = datetime.strptime(trange[0],self.tformat)
            t2 = datetime.strptime(trange[1],self.tformat)
            return [t1,t2]
        elif isinstance(trange[0],datetime):
            return trange
        else:
            raise Exception('unknown time format')

    def get_coord_names(self,varname):
        """
        Try to automatically workout the coordinate names

        This is probably not very robust
        """
        dims = self._nc.variables[varname].dimensions

        ndims = len(dims)

        try:
            coordinates = self._nc.variables[varname].coordinates

            # Dimensions are usually (time,depth,y,x)
            # Coordinates are generally (lon,lat,date)
            coordlist = coordinates.split()
            coordlist.reverse()

            timecoord, zcoord, ycoord0, xcoord0 = \
                dims[0],dims[1],coordlist[-2],coordlist[-1]
        except:
            if ndims==4:
                timecoord, zcoord, ycoord0 ,xcoord0 = dims
            elif ndims==3:
                 timecoord, ycoord0 ,xcoord0 = dims
                 zcoord=None
            elif ndims==2:
                 ycoord0 ,xcoord0 = dims
                 zcoord=None
                 timecoord=None # Used for writing


        # Do a hackish check to see if x and y are in the right order
        if 'lat' in xcoord0.lower() or 'lon' in ycoord0.lower():
            xcoord = ycoord0
            ycoord = xcoord0
        else:
            xcoord = xcoord0
            ycoord = ycoord0

        # Another hack...
        if xcoord=='X' or ycoord=='Y':
            xcoord='Longitude'
            ycoord='Latitude'
            
        return timecoord,xcoord,ycoord,zcoord

    def write_var(self,outfile,var,data):

        # Check if the file is open and/or exists
        if os.path.isfile(outfile):
            print('File exists - appending...')
            self._outnc = Dataset(outfile,'a')
        else:
            if '_outnc' not in self.__dict__:
                print('\tOpening %s'%outfile)
                self._outnc = Dataset(outfile, mode='w', \
                    format='NETCDF4_CLASSIC', data_model='NETCDF4_CLASSIC')

        ##
        if self.multifile:
            nc = self.MF.get_filename_only(var=var)
            self._nc = Dataset(nc)


        localvar = var
        remotevar = getattr(self,var)
        dapobj = getattr(self,'_%s'%var)

        # Get the coordinate variables
        timecoord,xcoord,ycoord,zcoord = \
            self.get_coord_names(remotevar)
     

        # Write the coordinate variables (will not be overwritten)
        self.create_ncvar_fromvarname(xcoord,dimsizes=dapobj.localX.shape)
        self._outnc.variables[xcoord][:]=dapobj.localX
        self.create_ncvar_fromvarname(ycoord,dimsizes=dapobj.localY.shape)
        self._outnc.variables[ycoord][:]=dapobj.localY

        is3d = False
        if not zcoord is None:
            self.create_ncvar_fromvarname(zcoord,dimsizes=dapobj.localZ.shape)
            self._outnc.variables[zcoord][:]=dapobj.localZ
            is3d = True

        # Write the time variable
        self.write_time_var(dapobj.localtime,timecoord)

        # Create the actual variable
        if is3d:
            dimsize = (None, dapobj.nz, dapobj.ny, dapobj.nx)
        else:
            dimsize = (None, dapobj.ny, dapobj.nx)

        if timecoord is None:
            create_time=True
        else:
            create_time=False

        self.create_ncvar_fromvarname(remotevar,dimsizes=dimsize, create_time=create_time)


        # Write to the variable
        self._outnc.variables[remotevar]\
            [self.tindex[0]:self.tindex[1],...] = data


    def write_time_var(self,tout,timecoord):
        """
        Write the time variale into the local file

        if it exists append the data into the right spot
        """
        # Check if local file has time variable
        if hasvar(self._outnc,timecoord):
            # Load the local time variable
            localtime = gettime(self._outnc,timecoord)

            # Get the index for insertion
            t = self._outnc.variables[timecoord]

            try:
                # This doesn't work if there is only one time value
                t1 = date2index(tout[0],t)
            except:
                if tout[0]>localtime[0]:
                    t1 = 1
                else:
                    t1 = 0
            self.tindex = [t1,t1+tout.shape[0]]
            print(self.tindex)

            # Check to see if we need to write
            if localtime.shape[0]<self.tindex[1]:
                self._outnc.variables[timecoord][self.tindex[0]:self.tindex[1]] = \
                    date2num(tout,t.units)

        else:
            # Create the variable (note unlimited dimension size)
            if timecoord is not None:
                self.create_ncvar_fromvarname(timecoord,dimsizes=(None,))
            else:
                # Create the time variable manually (necessary for 2D arrays)
                print('No time variable - creating one manually...')
                varname='time'
                if not hasdim(self._outnc,'time'):
                    tdim = self._outnc.createDimension('time', None)
                    dims = (tdim,)
                else:
                    tdim = self._outnc.dimensions['time']
                    dims = (tdim,)

                if not hasvar(self._outnc,'time'):
                    tmp=self._outnc.createVariable(varname, 'f8', ('time',))

                    # Copy the attributes
                    tmp.setncattr('long_name','time')
                    tmp.setncattr('units','seconds since 1970-01-01')

            if timecoord is None:
                timecoord='time'

            print('tout... ', tout)
            # Write the data (converts to netcdf units)
            t = self._outnc.variables[timecoord]
            self._outnc.variables[timecoord][:] = \
                date2num(tout,t.units)

            self.tindex = [0,tout.shape[0]]


    def create_gridfile_fromvar(self,outfile,varname):
        """
        Create a local file with the grid information from the variable
        """
        if os.path.isfile(outfile):
            print('File exists - appending...')
            self._outnc = Dataset(outfile,'a')
        else:
            if '_outnc' not in self.__dict__:
                print('\tOpening %s'%outfile)
                self._outnc = Dataset(outfile,'w')

        print('Creating local netcdf file with grid data...')

        # Get the variables to store in the grid file
        timecoord,xcoord,ycoord,zcoord = self.get_coord_names(varname)

        # Create the grid netcdf file
        self.create_ncfile(outfile)

        # Create the variables
        # Download the coordinate variable data
        if not zcoord==None:
            self.create_ncvar_fromvarname(zcoord)
            print('Downloading Z...')
            self._outnc.variables[zcoord][:] = self._nc.variables[zcoord][:]

        #X
        self.create_ncvar_fromvarname(xcoord)
        print('Downloading X...')
        self._outnc.variables[xcoord][:] = self._nc.variables[xcoord][:]

        #Y
        self.create_ncvar_fromvarname(ycoord)
        print('Downloading Y...')
        self._outnc.variables[ycoord][:] = self._nc.variables[ycoord][:]
        

    def create_ncvar_fromvarname(self,varname,dimsizes=None, create_time=False):
        """
        Create a netcdf variable using a remote variable name

        dimsizes can be passed as a list which will override the native grid

        'create_time': Adds a time dimensions to variables without one

        """

        if hasvar(self._outnc,varname):
            print('Variable exists - exiting.')
            return

        # List the dimensions
        dims = self._nc.variables[varname].dimensions
        for ii,dim in enumerate(dims):
            # Create the dimension if it doesn't exist
            if not hasdim(self._outnc,dim):
                if dimsizes is None:
                    dimsize = self._nc.dimensions[dim].__len__()
                else:
                    dimsize = dimsizes[ii]

                self._outnc.createDimension(dim,dimsize)

        print(varname)
        V = self._nc.variables[varname]
        # Create the variable
        dimensions = V.dimensions
        if create_time:
            dimensions = ('time',)+dimensions

        # Try and find the fillvalue
        fill_value = None
        if '_FillValue' in V.ncattrs():
            fill_value = getattr(V,'_FillValue')

        tmp= self._outnc.createVariable(varname, V.dtype, dimensions,
                zlib=True, fill_value=fill_value)

        # Copy the attributes
        for aa in V.ncattrs():
            if aa != '_FillValue':
                tmp.setncattr(aa,getattr(V,aa))

    def create_ncfile(self,ncfile):
        nc = Dataset(ncfile, mode='w',\
                data_model='NETCDF4_CLASSIC', format='NETCDF4_CLASSIC')
        nc.Title = '%s model data'%(self.type)
        nc.url = '%s'%(self.ncurl)
        
        self._outnc=nc

class MFncdap(object):
    """
    Multi-file class for opendap netcdf files
    
    MFDataset module is not compatible with opendap data 
    """
    
    timevar = 'time'
    tformat = '%Y%m%d.%H%M%S'

    # Not used here
    var = None
    
    def __init__(self,ncfilelist,**kwargs):
        
        self.__dict__.update(kwargs)
        self.ncfilelist = ncfilelist
        self.nfiles = len(self.ncfilelist)
        print('Retrieving the time information from files...')
        
        self._timelookup = {}
        timeall = []
        #self.time = np.zeros((0,))
        
        for f in self.ncfilelist:
            print(f)
            nc = Dataset(f)
            t = nc.variables[self.timevar]
            time = num2pydate(t[:].ravel(),t.units)#.tolist()
            nc.close()
            
            #self.timelookup.update({f:time})
            timeall.append(time)

            # Create a dictionary of time and file
            for ii,tt in enumerate(time):
                tstr = datetime.strftime(tt,self.tformat)
                if tstr in self._timelookup:
                    # Overwrite the old key
                    self._timelookup[tstr]=(f,ii)
                else:
                    self._timelookup.update({tstr:(f,ii)})

        #self.time = np.asarray(self.time)
        timeall = np.array(timeall)
        self.time = np.unique(timeall)
            
    def __call__(self,time, var=None):
        """
        Return the filenames and time index of the closest time
        """
        fname = []
        tind = []
        for t in time:
            tstr = datetime.strftime(t,self.tformat)
            f,ii = self._timelookup[tstr]
            fname.append(f)
            tind.append(ii)

        # Build a dictionary with each time slice in each bin
        # Order is important
        tslice_dict = collections.OrderedDict()
        for tt, f in zip(tind, fname):
            if f not in tslice_dict:
                tslice_dict.update({f:[]})
            
            tslice_dict[f].append(tt)


        for f in list(tslice_dict.keys()):
             vals = tslice_dict[f]
             tslice_dict[f] = [min(vals), max(vals)]

        return tind, fname, tslice_dict

    def get_timeslice(self, var=None):
        """
        Return the start and end time for each file
        """
        t0=[]
        t1=[]
        for ii in range(self.nfiles):
             tind, ncfiles = self.__call__(self.time[0], var=var)
             t0.append(tind[0])
             t1.append(tind[-1])

        return t0, t1

    def get_filename_only(self, var=None):
        """
        Returns the first file only
        """
        return self.ncfilelist[0]

 
