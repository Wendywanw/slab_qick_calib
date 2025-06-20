# -*- coding: utf-8 -*-
"""
:Authors: Phil Reinhold & David Schuster

The preferred format for saving data permanently is the
:py:class:`SlabFile`. This is a thin wrapper around the h5py_
interface to the HDF5_ file format. Using a SlabFile is much like
using a traditional python dictionary_, where the keys are strings,
and the values are `numpy arrays`_. A typical session using SlabFiles
in this way might look like this::

  import numpy as np
  from slab.datamanagement import SlabFile

  f = SlabFile('test.h5')
  f['xpts'] = np.linspace(0, 2*np.pi, 100)
  f['ypts'] = np.sin(f['xpts']) 
  f.attrs['description'] = "One period of the sine function"

Notice several features of this interaction.

1. Numpy arrays are inserted directly into the file by assignment, no function calls needed
2. Datasets are retrieved from the file and used as you would a numpy array
3. Non-array elements can be saved in the file with the aid of the 'attrs' dictionary

.. _numpy arrays: http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
.. _dictionary: http://docs.python.org/2/tutorial/datastructures.html#dictionaries
.. _HDF5: http://www.hdfgroup.org/HDF5/
.. _h5py: https://code.google.com/p/h5py/
"""

import numpy as np
import h5py
import inspect
import datetime
import json
import os.path
import copy



class h5File(h5py.File):
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)

    def add(self, key, data):
        data = np.array(data)
        try:
            self.create_dataset(key, shape=data.shape,
                                maxshape=tuple([None] * len(data.shape)),
                                dtype=str(data.dtype))
        except RuntimeError:
            del self[key]
            self.create_dataset(key, shape=data.shape,
                                maxshape=tuple([None] * len(data.shape)),
                                dtype=str(data.dtype))
        self[key][...] = data

    def append(self, key, data, forceInit=False):
        data = np.array(data)
        try:
            self.create_dataset(key, shape=tuple([1] + list(data.shape)),
                                maxshape=tuple([None] * (len(data.shape) + 1)),
                                dtype=str(data.dtype))
        except RuntimeError:
            if forceInit == True:
                del self[key]
                self.create_dataset(key, shape=tuple([1] + list(data.shape)),
                                    maxshape=tuple([None] * (len(data.shape) + 1)),
                                    dtype=str(data.dtype))
            dataset = self[key]
            Shape = list(dataset.shape)
            Shape[0] = Shape[0] + 1
            dataset.resize(Shape)

        dataset = self[key]
        try:
            dataset[-1, :] = data
        except TypeError:
            dataset[-1] = data
            # Usage require strictly same dimensionality for all data appended.
            # currently I don't have it setup to return a good exception, but should


class SlabFile(h5py.File):
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
       
        self.flush()

    # Methods for proxy use    
    def _my_ds_from_path(self, dspath):
        """returns the object (dataset or group) specified by dspath"""
        branch = self
        for ds in dspath:
            branch = branch[ds]
        return branch

    def _my_assign_dset(self, dspath, ds, val):
        print('assigning', ds, val)
        branch = self._my_ds_from_path(dspath)
        branch[ds] = val

    def _get_dset_array(self, dspath):
        """returns a pickle-safe array for the branch specified by dspath"""
        branch = self._my_ds_from_path(dspath)
        if isinstance(branch, h5py.Group):
            return 'group'
        else:
            return (H5Array(branch), dict(branch.attrs))

    def _get_attrs(self, dspath):
        branch = self._my_ds_from_path(dspath)
        return dict(branch.attrs)

    def _set_attr(self, dspath, item, value):
        branch = self._my_ds_from_path(dspath)
        branch.attrs[item] = value

    def _call_with_path(self, dspath, method, args, kwargs):
        branch = self._my_ds_from_path(dspath)
        return getattr(branch, method)(*args, **kwargs)

    def _ping(self):
        return 'OK'

    def set_range(self, dataset, xmin, xmax, ymin=None, ymax=None):
        if ymin is not None and ymax is not None:
            dataset.attrs["_axes"] = ((xmin, xmax), (ymin, ymax))
        else:
            dataset.attrs["_axes"] = (xmin, xmax)

    def set_labels(self, dataset, x_lab, y_lab, z_lab=None):
        if z_lab is not None:
            dataset.attrs["_axes_labels"] = (x_lab, y_lab, z_lab)
        else:
            dataset.attrs["_axes_labels"] = (x_lab, y_lab)

    def append_line(self, dataset, line, axis=0):
        if isinstance(dataset,str): dataset=str(dataset)
        if isinstance(dataset, str):
            try:
                dataset = self[dataset]
            except:
                shape, maxshape = (0, len(line)), (None, len(line))
                if axis == 1:
                    shape, maxshape = (shape[1], shape[0]), (maxshape[1], maxshape[0])
                self.create_dataset(dataset, shape=shape, maxshape=maxshape, dtype='float64')
                dataset = self[dataset]
        shape = list(dataset.shape)
        shape[axis] = shape[axis] + 1
        dataset.resize(shape)
        if axis == 0:
            dataset[-1, :] = line
        else:
            dataset[:, -1] = line
        self.flush()

    def append_pt(self, dataset, pt):
        if isinstance(dataset,str): dataset=str(dataset)
        if isinstance(dataset, str) :
            try:
                dataset = self[dataset]
            except:
                self.create_dataset(dataset, shape=(0,), maxshape=(None,), dtype='float64')
                dataset = self[dataset]
        shape = list(dataset.shape)
        shape[0] = shape[0] + 1
        dataset.resize(shape)
        dataset[-1] = pt
        self.flush()

    def append_dset_pt(self, dataset, pt):
        shape = dataset.shape[0]
        shape = shape + 1
        dataset.resize((shape, ))
        dataset[-1] = pt
        dataset.flush()

    def note(self, note):
        """Add a timestamped note to HDF file, in a dataset called 'notes'"""
        ts = datetime.datetime.now()
        try:
            ds = self['notes']
        except:
            ds = self.create_dataset('notes', (0,), maxshape=(None,), dtype=h5py.new_vlen(str))

        shape = list(ds.shape)
        shape[0] = shape[0] + 1
        ds.resize(shape)
        ds[-1] = str(ts) + ' -- ' + note
        self.flush()

    def get_notes(self, one_string=False, print_notes=False):
        """Returns notes embedded in HDF file if present.
        @param one_string=False if True concatenates them all together
        @param print_notes=False if True prints all the notes to stdout
        """
        try:
            notes = list(self['notes'])
        except:
            notes = []
        if print_notes:
            print('\n'.join(notes))
        if one_string:
            notes = '\n'.join(notes)
        return notes

    def add_data(self, f, key, data):
        data = np.array(data)
        try:
            f.create_dataset(key, shape=data.shape,
                             maxshape=tuple([None] * len(data.shape)),
                             dtype=str(data.dtype))
        except RuntimeError:
            del f[key]
            f.create_dataset(key, shape=data.shape,
                             maxshape=tuple([None] * len(data.shape)),
                             dtype=str(data.dtype))
        f[key][...] = data

    def append_data(self, f, key, data, forceInit=False):
        """
        the main difference between append_pt and append is thta
        append takes care of highier dimensional data, but not append_pt
        """

        data = np.array(data)
        try:
            f.create_dataset(key, shape=tuple([1] + list(data.shape)),
                             maxshape=tuple([None] * (len(data.shape) + 1)),
                             dtype=str(data.dtype))
        except RuntimeError:
            if forceInit == True:
                del f[key]
                f.create_dataset(key, shape=tuple([1] + list(data.shape)),
                                 maxshape=tuple([None] * (len(data.shape) + 1)),
                                 dtype=str(data.dtype))
            dataset = f[key]
            Shape = list(dataset.shape)
            Shape[0] = Shape[0] + 1
            dataset.resize(Shape)

        dataset = f[key]
        try:
            dataset[-1, :] = data
        except TypeError:
            dataset[-1] = data
            # Usage require strictly same dimensionality for all data appended.
            # currently I don't have it setup to return a good exception, but should

    def add(self, key, data):
        self.add_data(self, key, data)

    def append(self, dataset, pt):
        self.append_data(self, dataset, pt)

    # def save_script(self, name="_script"):
    # self.attrs[name] = get_script()
    def save_dict(self, dict, group='/'):
        if group not in self:
            self.create_group(group)
        for k in list(dict.keys()):
            self[group].attrs[k] = dict[k]

    def get_dict(self, group='/'):
        d = {}
        g=self[group]
        for k in g.attrs:
            d[k] = g.attrs[k]
        return d

    get_attrs = get_dict
    save_attrs = save_dict

    def get_group_data(self, group='/'):
        data={'attrs': self.get_dict(group)}
        
        g=self[group]
        for k in g.keys():
            data[k]=np.array(g[k])
        return data

    def save_settings(self, dic, group='settings'):
        self.save_dict(dic, group)

    def load_settings(self, group='settings'):
        return self.get_dict(group)

    def load_config(self):
        if 'config' in list(self.attrs.keys()):
            return AttrDict(json.loads(self.attrs['config']))
        else:
            return None
        


def set_range(dset, range_dsets, range_names=None):
    """
    usage:
        ds['x'] = linspace(0, 10, 100)
        ds['y'] = linspace(0, 1, 10)
        ds['z'] = [ sin(x*y) for x in ds['x'] for y in ds['y'] ]
        set_range(ds['z'], (ds['x'], ds['y']), ('x', 'y'))
    """
    for i, range_ds in enumerate(range_dsets):
        dset.dims.create_scale(range_ds)
        dset.dims[i].attach_scale(range_ds)
        if range_names:
            dset.dims[i].label = range_names[i]


def get_script():
    """returns currently running script file as a string"""
    fname = inspect.stack()[-1][1]
    if fname == '<stdin>':
        return fname
    # print fname
    f = open(fname, 'r')
    s = f.read()
    f.close()
    return s


def open_to_path(h5file, path, pathsep='/'):
    f = h5file
    for name in path.split(pathsep):
        if name:
            f = f[name]
    return f


def get_next_trace_number(h5file, last=0, fmt="%03d"):
    i = last
    while (fmt % i) in h5file:
        i += 1
    return i


def open_to_next_trace(h5file, last=0, fmt="%03d"):
    return h5file[fmt % get_next_trace_number(h5file, last, fmt)]


def load_array(f, array_name):
    if f[array_name].len() == 0:
        a = []
    else:
        a = np.zeros(f[array_name].shape)
        f[array_name].read_direct(a)

    return a

def load_slabfile_data(fname, path='', group='/'):
    fullname=os.path.join(path, fname)
    with SlabFile(fullname, 'r') as f:
        data=f.get_group_data(group)
    return data


class AttrDict(dict):
    def __init__(self, value=None):
        super().__init__()
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        super(AttrDict,self).__setitem__(key, value)


    def __getitem__(self, key):
        v=super().__getitem__(key)
        if isinstance(v, dict) and not isinstance (v, AttrDict):
            return AttrDict(v)
        else:
            return v

    def __setattr__(self, a ,v):
        return self.__setitem__(a,v)
    def __getattr__(self, a):
        if a in self:
            return self.__getitem__(a)
        else:
            return self.__getattribute__(a) #@IgnoreException
        
    def to_dict(self):
        d={}
        for k,v in self.items():
            if isinstance(v, AttrDict):
                d[k]=v.to_dict()
            else:
                d[k]=v
        return d
    
    # def __deepcopy__(self, memo):
    #     # Deepcopy only the id attribute, then construct the new instance and map
    #     # the id() of the existing copy to the new instance in the memo dictionary
    #     memo[id(self)] = newself = self.__class__(copy.deepcopy(self.id, memo))
    #     # Now that memo is populated with a hashable instance, copy the other attributes:
    #     newself.degree = copy.deepcopy(self.degree, memo)
    #     # Safe to deepcopy edge_dict now, because backreferences to self will
    #     # be remapped to newself automatically
    #     newself.edge_dict = copy.deepcopy(self.edge_dict, memo)
    #     return newself

    # def __new__(cls, p_id):
    #     self = super().__new__(cls)  # Must explicitly create the new object
    #     # Aside from explicit construction and return, rest of __new__
    #     # is same as __init__
    #     self.id = p_id
    #     self.edge_dict = {}
    #     self.degree = 0
    #     return self  # __new__ returns the new object

    # def __getnewargs__(self):
    #     # Return the arguments that *must* be passed to __new__
    #     return (self.id,)

   