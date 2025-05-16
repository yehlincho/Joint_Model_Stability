## originally from https://files.ipd.uw.edu/krypton/TrRosetta/design/to_pdb.py

import numpy as np
from sklearn.manifold import MDS
from utils import *

def metric_MDS(x):
  return MDS(3,dissimilarity="precomputed").fit_transform(x)

def classic_MDS(M,dims=3):
  L = M.shape[0]
  d = np.square(M)
  c = np.eye(L) - np.ones((L,L))/L
  m = -0.5 * c @ d @ c
  s,u = np.linalg.eigh(m)
  X = u[:,-dims:] * np.sqrt(s[-dims:])
  return X

def fix_chirality(x):
  n,ca,c = x.reshape(-1,3,3).transpose(1,0,2)
  phi_mu = to_dih(c[:-1],n[1:],ca[1:],c[1:]).mean()
  if phi_mu > 0: return x*[1,1,-1]
  else: return x

def vals_to_xyz(theta, phi, dist, omega, mask=None, verbose=False, mds="metric"):
  include=["n","a","c"]
  at = len(include)
  L = dist.shape[0]

  def flat(x):
    np.fill_diagonal(x,np.nan)
    return x.flatten()[:,None]

  d = {"na":1.458,"an":1.458,
       "ac":1.523,"ca":1.523,
       "ab":1.522,"ba":1.522,
       "c_n":1.329,
       "nb":2.447,"bn":2.447,
       "cb":2.499,"bc":2.499,
       "nc":2.460,"cn":2.460,
       "anc":0.615,"banc":-2.143,
       "nab":1.927,"ban":1.927,
       "nabb":flat(theta),"bban":flat(theta.T),
       "abb":flat(phi),"bba":flat(phi.T),
       "bb":flat(dist),"abba":flat(omega)}

  # reconstruct pairwise coordinates
  x,y = {},{}
  x["n"] = np.array([[0,0,0]])
  x["a"] = np.array([[0,0,d["na"]]])
  x["b"] = np.array([[0,d["ab"]*np.sin(d["nab"]),d["na"]-d["ab"]*np.cos(d["nab"])]])
  x["c"] = extend(x["b"],x["a"],x["n"],d["nc"],d["anc"],d["banc"])
  y["b"] = extend(x["n"],x["a"],x["b"],d["bb"],d["abb"],d["nabb"])
  y["a"] = extend(x["a"],x["b"],y["b"],d["ba"],d["bba"],d["abba"])
  y["n"] = extend(x["b"],y["b"],y["a"],d["an"],d["ban"],d["bban"])
  y["c"] = extend(y["b"],y["a"],y["n"],d["nc"],d["anc"],d["banc"])

  # get backbone phi/psi/omega
  yn = y["n"][1::(L+1)]
  ya = y["a"][1::(L+1)]
  yc = y["c"][1::(L+1)]
  psi = to_dih(x["n"],x["a"],x["c"],yn)
  omg = to_dih(x["a"],x["c"],yn,ya)
  phi = to_dih(x["c"],yn,ya,yc)

  # compute distance between coordinates
  dm = []
  for i in (include):
    for j in (include):
      dm_ = to_len(x[i],y[j]).reshape(L,L)
      if i == j: np.fill_diagonal(dm_,0)
      else: np.fill_diagonal(dm_,d[i+j])
      dm.append(dm_)

  sparse_dm = np.array(dm).reshape(at,at,L,L).transpose(2,0,3,1).reshape(at*L,at*L)
  if mask is not None:
    mask = np.tile(mask[:,None,:,None],[1,at,1,at]).reshape(at*L,at*L)
    sparse_dm[mask == 0] = np.inf

  # fix bond lengths
  def fill_diag(x,k,d):
    np.fill_diagonal(x[k:,:],d)
    np.fill_diagonal(x[:,k:],d)

  bonds = [d["na"],d["ac"],d["c_n"]]
  fill_diag(sparse_dm,1,bonds)
  np.fill_diagonal(sparse_dm,0.0)

  # fill-in missing distances
  full_dm = FW(sparse_dm)
  fill_diag(full_dm,1,bonds)

  # recover coordinates from dihedrals/distances
  fL,fA,fD = ppo_to_xlad(psi,omg,phi)
  xyz = metric_MDS(full_dm) if mds == "metric" else classic_MDS(full_dm)
  xyz = fix_chirality(xyz)
  xyz = refine(full_dm, xyz, W=mask, verbose=verbose)

  # refine atoms
  N,CA,C = xyz.reshape(L,-1,3).transpose(1,0,2)

  # place CB and O atoms
  CB = extend(C,N,CA,1.522,1.927,-2.143)
  O = extend(np.roll(N,-1,axis=0),CA,C,1.231,2.108,-3.142)
  CA_dm = full_dm[1::3,:][:,1::3]
  return np.stack([N,CA,C,O,CB],1), CA_dm

def bins_to_vals(theta, phi, dist, omega):
  def get_bins(s,e,b):
    a = np.linspace(s,e,b+1)
    return (a[1:]+a[:-1])/2

  def bin2val(a, b, dih=False, eps=1e-8):
    a_sub = a[...,1:]
    if dih: return np.arctan2(bin2val(a,np.sin(b)),bin2val(a,np.cos(b)))
    else: return (a_sub*b).sum(-1)/(a_sub.sum(-1)+eps)

  # get mask
  mask = dist[...,1:].max(-1)
  mask[dist.argmax(-1) == 0] = 0

  # convert bins to values
  len_bins = get_bins(2,20,36)
  ang_bins = get_bins(0,np.pi,12)
  dih_bins = get_bins(-np.pi,np.pi,24)

  dist = bin2val(dist,len_bins)
  phi = bin2val(phi,ang_bins)
  omega = bin2val(omega,dih_bins,dih=True)
  theta = bin2val(theta,dih_bins,dih=True)

  return theta, phi, dist, omega, mask

def ppo_to_xlad(PSI,OMG,PHI):
  d = {"na":1.458,"ac":1.523,"cn":1.329,
       "nac":1.941,"acn":2.028,"cna":2.124}
  # initialize N,CA,C
  ln = PSI.shape[0]
  L = np.tile([d["cn"],d["na"],d["ac"]],ln)
  L = np.append([d["na"],d["ac"]],L)
  A = np.tile([d["acn"],d["cna"],d["nac"]],ln)
  A = np.append([d["nac"]],A)
  D = np.array([PSI,OMG,PHI]).T.flatten()
  return L,A,D

def refine(dm, xyz, W=None, iter=100, verbose=False):
  if W is None: W = np.ones_like(dm)
  rms = lambda: np.sqrt((W*np.square(dm-to_len_pw(xyz))).sum()/W.sum())
  for p in range(iter):
    # compute gradient
    x_x = (xyz[:,None] - xyz[None,:]).T
    dm_ = np.sqrt(np.square(x_x).sum(0)+1e-8)
    g = (2.0 * x_x * (dm - dm_))/dm_
    g = ((g*W).sum(-1)/W.sum(-1)).T
    # apply gradient
    xyz = xyz - g
    if verbose: print(p,rms())
  return xyz

def FW(d):
  for m in range(d.shape[0]):
    o = d[m]
    d = np.minimum(d, o[None,:]+o[:,None])
  return d

def feat_to_xyz(feat, mds="metric"):
  D = split_feat(feat)
  return vals_to_xyz(*bins_to_vals(D["theta"],D["phi"],D["dist"],D["omega"]), mds=mds)

def save_PDB(pdb_out, coords, dm, seq):
  atoms = ['N','CA','C','O','CB']
  error = np.sqrt(np.square(dm-to_len_pw(coords[:,1])).mean(-1))
  out = open(pdb_out,"w")
  k = 0
  for r,residue in enumerate(coords):
    AA = aa_1_3[seq[r]]
    for a,atom in enumerate(residue):
      if AA == "GLY" and atoms[a] == "CB": continue
      x,y,z = atom
      out.write("ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                % (k+1,atoms[a],AA,"A",r+1,x,y,z,1,error[r]))
      k += 1
  out.close()
