import numpy as np
import numpy.linalg as lg
import os


def sample_cylinder(npts=2048):
    l = np.random.rand()
    r = np.random.rand()

    a1 = 2 * np.pi * r * r
    a2 = 2 * np.pi * r * l
    suma = a1 + a2
    p = a1 / suma

    nptscirc = int(np.floor(p * npts/2))
    nptscyl = npts - 2*nptscirc

    u = np.random.rand(nptscirc,2)
    u[:,0] = np.sqrt(u[:,0])
    x1 = np.concatenate((-l/2*np.ones(nptscirc), u[:,0]* r * np.cos(2 * np.pi * u[:,1]), u[:,0] * r * np.sin(2 * np.pi * u[:,1]))).reshape((nptscirc, 3), order='F')

    u = np.random.rand(nptscirc,2)
    u[:,0] = np.sqrt(u[:,0])
    x2 = np.concatenate((l/2*np.ones(nptscirc), u[:,0] * r* np.cos(2 * np.pi * u[:,1]), u[:,0] * r * np.sin(2 * np.pi * u[:,1]))).reshape((nptscirc, 3), order='F')

    u = np.random.rand(nptscyl,2)
    x3 = np.concatenate((-l/2+l*u[:,0], r * np.cos(2 * np.pi * u[:,1]), r * np.sin(2 * np.pi * u[:,1]))).reshape((nptscyl, 3), order='F')

    x = np.concatenate((x1,x2,x3), axis = 0)
    return x


def sample_rectangle(npts=2048):
    lx = np.random.rand()
    ly = np.random.rand()
    lz = np.random.rand()

    ax = ly*lz
    ay = lx*lz
    az = lx*ly
    suma = ax+ay+az
    ax = ax / suma
    ay = ay / suma
 
    nptsyz = int(np.floor(ax*npts))
    nptsxz = int(np.floor(ay*npts))
    nptsxy = npts - nptsyz - nptsxz


    n = int(np.floor(nptsyz/2))
    u = np.random.rand(n,2) - 0.5
    x1 = np.concatenate((-lx/2*np.ones((n)), ly*u[:,0], lz*u[:,1])).reshape((n,3), order='F')

    u = np.random.rand(nptsyz - n,2)-0.5
    x2 = np.concatenate((lx/2*np.ones(nptsyz - n), ly*u[:,0], lz*u[:,1])).reshape((nptsyz - n, 3),order='F')

    

    n = int(np.floor(nptsxz/2))
    u = np.random.rand(n,2)-0.5
    x3 = np.concatenate((lx*u[:,0], -ly/2*np.ones(n), lz*u[:,1])).reshape((n,3),order='F')


    u = np.random.rand(nptsxz - n,2)-0.5
    x4 = np.concatenate((lx*u[:,0], ly/2*np.ones(nptsxz - n), lz*u[:,1])).reshape((nptsxz -n, 3),order='F')



    n = int(np.floor(nptsxy/2))
    u = np.random.rand(n,2)-0.5
    x5 = np.concatenate((lx*u[:,0], ly*u[:,1], -lz/2*np.ones(n))).reshape((n,3),order='F')


    u = np.random.rand(nptsxy - n,2)-0.5
    x6 = np.concatenate((lx*u[:,0], ly*u[:,1], lz/2*np.ones(nptsxy - n))).reshape((nptsxy -n, 3),order='F')


    x = np.concatenate((x1,x2,x3,x4,x5,x6), axis = 0)
    return x


def sample_torus(npts=2048):
    r1 = np.random.rand()
    r2 = np.random.rand()

    if r1 < r2:
        a = r1
        r1 = r2
        r2 = a

    u = 2 * np.pi * np.random.rand(npts,1)
    v = 2 * np.pi * np.random.rand(npts,1)

    x = (r1 + r2*np.cos(v))*np.cos(u)
    y = (r1 + r2*np.cos(v))*np.sin(u)
    z = r2 * np.sin(v)

    return np.concatenate((x,y,z), axis = 1)


def normalize(x):
    mean = np.mean(x, axis = 0)
    x = x - mean
    mini = np.min(x)
    maxi = np.max(x)
    x = x / np.max([-mini, maxi])
    return x

def apply_random_rotation(x, dim=3):

    u = np.random.randn(3,1)
    u = u / lg.norm(u)

    v = np.random.randn(3,1)
    v = v - np.dot(u.transpose(1, 0), v) * u
    v = v / lg.norm(v)

    w = np.cross(u, v, axis=0)
    w = w / lg.norm(w)

    M = np.concatenate((u,v,w), axis=1)

    x = np.matmul(x,M)
    x = normalize(x)
    return x


def main():
    npts = 2048
    nshapes = 500
    nshapestest = 300

    ndigits = len(str(nshapes))

#training data
    if not os.path.exists('./data/train/00/'):
        os.makedirs('./data/train/00/')
    if not os.path.exists('./data/train/01/'):
        os.makedirs('./data/train/01/')
    if not os.path.exists('./data/train/02/'):
        os.makedirs('./data/train/02/')

    
    for i in range(nshapes):
        x = sample_cylinder(npts)
        x = apply_random_rotation(x)
        np.savetxt(f"./data/train/00/{str(i).zfill(ndigits)}.asc", x)
        x = sample_rectangle(npts)
        x = apply_random_rotation(x)
        np.savetxt(f"./data/train/01/{str(i).zfill(ndigits)}.asc", x)

        x = sample_torus(npts)
        x = apply_random_rotation(x)
        np.savetxt(f"./data/train/02/{str(i).zfill(ndigits)}.asc", x)

#testing data
    if not os.path.exists('./data/test/00/'):
        os.makedirs('./data/test/00/')
    if not os.path.exists('./data/test/01/'):
        os.makedirs('./data/test/01/')
    if not os.path.exists('./data/test/02/'):
        os.makedirs('./data/test/02/')

    
    for i in range(nshapestest):
        x = sample_cylinder(npts)
        x = apply_random_rotation(x)
        np.savetxt(f"./data/test/00/{str(i).zfill(ndigits)}.asc", x)
        x = sample_rectangle(npts)
        x = apply_random_rotation(x)
        np.savetxt(f"./data/test/01/{str(i).zfill(ndigits)}.asc", x)

        x = sample_torus(npts)
        x = apply_random_rotation(x)
        np.savetxt(f"./data/test/02/{str(i).zfill(ndigits)}.asc", x)

main()
