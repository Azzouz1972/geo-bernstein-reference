import zipfile, pickle, io, struct
import numpy as np
from collections import OrderedDict

class _Stor:
    def __init__(self, key, dtype): self.key=key; self.dtype=dtype

def load_state_dict_numpy(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split('/')[0]
    raw = {}
    for n in z.namelist():
        if '/data/' in n:
            raw[n.split('/data/')[1]] = z.read(n)

    dtmap = {'FloatStorage':np.float32,'DoubleStorage':np.float64,
             'HalfStorage':np.float16,'LongStorage':np.int64,'IntStorage':np.int32}

    def rebuild_tensor_v2(storage, storage_offset, size, stride, *a):
        buf = raw[storage.key]
        arr = np.frombuffer(buf, dtype=storage.dtype)
        size=tuple(size); stride=tuple(stride)
        if len(size)==0: return arr[storage_offset]
        itemsize=arr.dtype.itemsize
        return np.lib.stride_tricks.as_strided(
            arr[storage_offset:], shape=size,
            strides=tuple(s*itemsize for s in stride)).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod=='torch._utils' and name=='_rebuild_tensor_v2': return rebuild_tensor_v2
            if mod=='torch._utils' and name=='_rebuild_parameter':
                return lambda data,*a: data
            if mod=='collections' and name=='OrderedDict': return OrderedDict
            if mod=='torch' and name.endswith('Storage'):
                dt=dtmap.get(name,np.float32)
                return ('STOR', dt)
            return lambda *a, **k: None
        def persistent_load(self, pid):
            # pid = ('storage', storage_type, key, location, numel)
            _, stype, key, *_ = pid
            dt = stype[1] if isinstance(stype, tuple) else np.float32
            return _Stor(key, dt)
    sd = U(io.BytesIO(z.read(root+'/data.pkl'))).load()
    return sd

if __name__=='__main__':
    import sys
    sd=load_state_dict_numpy('/home/claude/out/Geo-Bernstein_seed0.pt')
    for k,v in sd.items():
        print(f'{k:18} {tuple(np.shape(v))}  dtype={np.asarray(v).dtype}')
