
import gpt as g
import numpy as np
import cupy as cp #! For PyQUDA

# load pyquda modules
from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma
from pyquda_utils.core import X, Y, Z, T
from opt_einsum import contract

GEN_SIMD_WIDTH = 64


#ordered list of gamma matrix identifiers, needed for the tag in the correlator output
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
#my_proton_proj = ["P+","P+_Sz+","P+_Sx+","P+_Sx-"]
#my_proton_proj = ["P+"]

ordered_list_of_gammas = [g.gamma[5], g.gamma["T"], g.gamma["T"]*g.gamma[5],
                                      g.gamma["X"], g.gamma["X"]*g.gamma[5], 
                                      g.gamma["Y"], g.gamma["Y"]*g.gamma[5],
                                      g.gamma["Z"], g.gamma["Z"]*g.gamma[5], 
                                      g.gamma["I"], g.gamma["SigmaXT"], 
                                      g.gamma["SigmaXY"], g.gamma["SigmaXZ"], 
                                      g.gamma["SigmaZT"]
                            ]


#!/usr/bin/env python3
#
# GPT inversion sources selection
#
import h5py


def srcLoc_distri_eq(L, src_origin):
    source_positions = []
    i_src = 0
    div = 4
    for i in range(div):
        for j in range(div):
            for k in range(div):
                for l in range(div):
                    source_positions += [[round(i*L[0]/div+src_origin[0])%L[0], round(j*L[1]/div+src_origin[1])%L[1], round(k*L[2]/div+src_origin[2])%L[2], round(l*L[3]/div+src_origin[3])%L[3]]]
    return source_positions


def get_fwPropagator_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat)
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/frw_prop/frw_prop" + "." + cfg_tag + "." + ama_tag + "." + sm_tag + "." + src_tag

def get_c2pt_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".c2pt"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/c2pt/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_qTMDWF_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMDWF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMDWF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_qTMD_file_tag(data_dir, lat, cfg, ama,src, sm):
    
    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMD"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMD/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag
    
def get_qDA_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qDA"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qDA/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_qTMDWF_wallsrc_file_tag(data_dir, lat, cfg, ama, src, sm, quarkmom1, quarkmom2):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMDWF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    mom1_tag = "qx"+str(quarkmom1[0]) + "qy"+str(quarkmom1[1]) + "qz"+str(quarkmom1[2])
    mom2_tag = "qx"+str(quarkmom2[0]) + "qy"+str(quarkmom2[1]) + "qz"+str(quarkmom2[2])
    sm_tag  = str(sm)

    return data_dir + "/qTMDWF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag + ".fw_" + mom1_tag + ".bw_" + mom2_tag

def get_softFF_file_tag(data_dir, lat, cfg, ama, src, sm, quarkmom1, quarkmom2):
    
    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".softFF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    mom1_tag = "qx"+str(quarkmom1[0]) + "qy"+str(quarkmom1[1]) + "qz"+str(quarkmom1[2])
    mom2_tag = "qx"+str(quarkmom2[0]) + "qy"+str(quarkmom2[1]) + "qz"+str(quarkmom2[2])
    sm_tag  = str(sm)

    return data_dir + "/ff/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + ".fw_" + mom1_tag + ".bw_" + mom2_tag

def get_sample_log_tag(ama, src, sm):

    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    log_sample = ama_tag + "_" + src_tag + "_" + sm_tag

    return log_sample

def save_fwPropagator_hdf5(prop, tag, sm="SP"):

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')
    sm = f.create_group(f"prop/{sm}")
    for c in range(0, 3):
        for d in range(0, 4):
            dataset_tag = f"d{d}_c{c}"
            sm.create_dataset(dataset_tag, data=prop[:,:,:,:,:,d,:,c])
    f.close()

def save_proton_c2pt_hdf5(corr, tag, gammalist, plist):

    roll = -int(tag.split(".")[4].split('t')[1])

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')
    sm = f.create_group("SS")
    for ig, gm in enumerate(gammalist):
        g = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            #print('DEBUG:', np.shape(corr), np.shape(gammalist), ig, ip)
            g.create_dataset(dataset_tag, data=np.roll(corr[ig][ip], roll, axis=0))
    f.close()

def save_c2pt_hdf5(corr, tag, gammalist, plist, sm="SS"):

    # corr[link][plist][gammalist][t]
    roll = -int(tag.split(".")[4].split('t')[1])

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')
    sm = f.create_group(sm)
    for ig, gm in enumerate(gammalist):
        g = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g.create_dataset(dataset_tag, data=np.roll(corr[0][ip][ig], roll, axis=0))
    f.close()

def save_softFF_hdf5(corr, tag, pion_src, pion_sink, Gamma1, Gamma2, bT_dir, bT_length, tseplist):
    """
    bdir: direction of bT
    bT: length of bT
    """

    roll = -int(tag.split(".")[4].split('t')[1])
    bT_list = ['bX', 'bY']

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')

    keys_src = list(pion_src.keys())
    keys_sink = list(pion_sink.keys())
    keys_gm1 = list(Gamma1.keys())
    keys_gm2 = list(Gamma2.keys())
    for i in range(len(keys_src)):  # both src and sink have the same number of keys
        g_src = f.create_group(f"src{keys_src[i]}_sink{keys_sink[i]}")
        for j in range(len(keys_gm1)):
            g_gm = g_src.create_group(f"{keys_gm1[j]}_{keys_gm2[j]}")
            for k, dir in enumerate(bT_dir):
                for bT in range(0, bT_length+1):
                    g_bT = g_gm.create_group(bT_list[dir]+'_'+str(bT))
                    for its, ts in enumerate(tseplist):
                        g_bT.create_dataset(f'ts{str(ts)}', data=np.roll(corr[its][i][j][k][bT], roll, axis=0))
    f.close()

def save_qTMDWF_hdf5_subset(corr, tag, gammalist, plist, W_index_list, i_sub):

    roll = -int(tag.split(".")[4].split('t')[1])
    bT_list = ['b_X', 'b_Y']

    if g.rank() == 0:
        print("-->>",W_index_list)

    save_h5 = tag + ".h5"
    if i_sub == 0:
        f = h5py.File(save_h5, 'w')
    else:
        f = h5py.File(save_h5, 'a')
    sm = f.require_group("SP")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                #if g.rank() == 0 and ig == 0 and ip == 0:
                #    #g_p.keys()
                #    #g_data.keys()
                #    print("Want to save", path+'bz'+str(idx[1]))
                g_data.create_dataset('bz'+str(idx[1]), data=np.roll(corr[i][ip][ig], roll, axis=0))
    f.close()

def save_qTMDWF_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list):

    bT_list = ['b_X', 'b_Y']

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')

    sm = f.require_group("SP")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                g_data.create_dataset('bz'+str(idx[1]), data=corr[i][ip][ig])
    f.close()

def save_qTMDWF_hdf5(corr, tag, gammalist, plist, eta, b_T, b_z, bT_dir = [0,1]):

    roll = -int(tag.split(".")[4].split('t')[1])
    td_offset = b_T*b_z*len(eta)
    eta_offset = b_T*b_z
    bz_offset = b_T
    bT_list = ['b_X', 'b_Y']

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')
    sm = f.create_group("SP")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.create_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.create_group(p_tag)
            for transverse_direction in bT_dir:
                g_T = g_p.create_group(bT_list[transverse_direction])
                for eta_idx, current_eta in enumerate(eta):
                    g_eta = g_T.create_group('eta'+str(current_eta))
                    for current_b_T in range(0, b_T):
                        g_bT = g_eta.create_group('bT'+str(current_b_T))
                        for current_bz in range(0, b_z):
                            bz_tag = 'bz'+str(current_bz)
                            W_index = current_b_T + bz_offset*current_bz + eta_offset*eta_idx + td_offset*transverse_direction
                            g_bT.create_dataset(bz_tag, data=np.roll(corr[W_index][ip][ig], roll, axis=0))
    f.close() 

# W_index_list[bT, bz, eta, Tdir]
def save_qTMD_proton_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list, tsep):

    bT_list = ['b_X', 'b_Y']

    #g.message("-->>",W_index_list)

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')

    g.message(f"no roll")
    g.message(f"corr.shape, {np.shape(corr)}")
    g.message(f"plist.shape, {np.shape(plist)}")
    sm = f.require_group("SS")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                g_data.create_dataset('bz'+str(idx[1]), data=corr[i][ip][ig][:tsep+2])
    f.close()

# W_index_list[bT, bz, eta, Tdir]
def save_qTMD_proton_hdf5(corr, tag, gammalist, plist, W_index_list, tsep):
    
    roll = -int(tag.split(".")[6].split('t')[1]) # 6: xyzt
    bT_list = ['b_X', 'b_Y']
 
    #g.message("-->>",W_index_list)

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')

    g.message(f"roll {roll}")
    g.message(f"corr.shape, {np.shape(corr)}")
    g.message(f"plist.shape, {np.shape(plist)}")
    sm = f.require_group("SS")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                #if g.rank() == 0:
                #    g.message("Want to save", path+'bz'+str(idx[1]))
                g_data.create_dataset('bz'+str(idx[1]), data=np.roll(corr[i][ip][ig], roll, axis=0)[:tsep+2])
    f.close()

# W_index_list[bT, bz, eta, Tdir]
def save_qTMD_proton_hdf5_subset(corr, tag, gammalist, plist, W_index_list, i_sub, tsep):

    roll = -int(tag.split(".")[6].split('t')[1]) # 6: xyzt
    bT_list = ['b_X', 'b_Y']

    g.message("-->>",W_index_list)

    save_h5 = tag + ".h5"
    if i_sub == 0:
        f = h5py.File(save_h5, 'w')
    else:
        f = h5py.File(save_h5, 'a')

    sm = f.require_group("SS")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                #g.message("Want to save", path+'bz'+str(idx[1]))
                g_data.create_dataset('bz'+str(idx[1]), data=np.roll(corr[i][ip][ig], roll, axis=0)[:tsep+2])
    f.close()



def uud_two_point(Q1, Q2, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q2))
    return g(g.color_trace(g.spin_trace(dq) * Q1 + dq * Q1))

def proton_contr(Q1, Q2):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    #Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
    corr = []
    for ig, gm in enumerate(ordered_list_of_gammas):
        Pp = gm
        corr += [g(g.trace(uud_two_point(Q1, Q2, Gamma) * Pp))]
    return corr
    #return g(g.trace(uud_two_point(Q1, Q2, Gamma) * Pp))

class proton_measurement:
    def __init__(self, parameters):
        self.plist = parameters["plist"]
        self.pol_list = ["P+_Sz+","P+_Sx+","P+_Sx-"]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]

    def set_input_facilities(self, corr_file):
        self.input_correlator = g.corr_io.reader(corr_file)

    def set_output_facilities(self, corr_file, prop_file):
        self.output_correlator = g.corr_io.writer(corr_file)
        
        if(self.save_propagators):
            self.output = g.gpt_io.writer(prop_file)

    def propagator_input(self, prop_file):
        g.message(f"Reading propagator file {prop_file}")
        read_props = g.load(prop_file)
        return read_props

    def propagator_output_k0(self, tag, prop_f):

        g.message("Saving forward propagator")
        prop_f_tag = "%s/%s" % (tag, str(self.pos_boost))
        self.output.write({prop_f_tag: prop_f})
        self.output.flush()
        g.message("Propagator IO done")

    def propagator_output(self, tag, prop_f, prop_b):

        g.message("Saving forward propagator")
        prop_f_tag = "%s/%s" % (tag, str(self.pos_boost)) 
        self.output.write({prop_f_tag: prop_f})
        self.output.flush()
        g.message("Saving backward propagator")
        prop_b_tag = "%s/%s" % (tag, str(self.neg_boost))
        self.output.write({prop_b_tag: prop_b})
        self.output.flush()
        g.message("Propagator IO done")

    def make_24D_inverter(self, U, evec_file):

        l_exact = g.qcd.fermion.zmobius(
            #g.convert(U, g.single),
            U,
            {
                "mass": 0.00107,
                "M5": 1.8,
                "b": 1.0,
                "c": 0.0,
                "omega": [
                    1.0903256131299373,
                    0.9570283702230611,
                    0.7048886040934104,
                    0.48979921782791747,
                    0.328608311201356,
                    0.21664245377015995,
                    0.14121112711957107,
                    0.0907785101745156,
                    0.05608303440064219 - 0.007537158177840385j,
                    0.05608303440064219 + 0.007537158177840385j,
                    0.0365221637144842 - 0.03343945161367745j,
                    0.0365221637144842 + 0.03343945161367745j,
                ],
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],
            },
        )

        l_sloppy = l_exact.converted(g.single)
        g.message(f"Loading eigenvectors from {evec_file}")
        g.mem_report(details=False)
        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)

        g.mem_report(details=False)
        pin = g.pin(eig[1], g.accelerator)
        g.message("creating deflated solvers")

        g.message("creating deflated solvers")
        light_innerL_inverter = g.algorithms.inverter.preconditioned(
           g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
           g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        light_innerH_inverter = g.algorithms.inverter.preconditioned(
            g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
            g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-4, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=12,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-4,
            maxiter=12,
        )


        ############### final inverter definitions
        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin
            

    def make_64I_inverter(self, U, evec_file):
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #64I params
                "mass": 0.000678,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],
                },

        )

        l_sloppy = l_exact.converted(g.single)
        g.message(f"Loading eigenvectors from {evec_file}")
        g.mem_report(details=False)
        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)

        g.mem_report(details=False)
        pin = g.pin(eig[1], g.accelerator)
        g.message("creating deflated solvers")

        light_innerL_inverter = g.algorithms.inverter.preconditioned(
           g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
           g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        light_innerH_inverter = g.algorithms.inverter.preconditioned(
            g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
            g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-4, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=12,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-4,
            maxiter=12,
        )


        ############### final inverter definitions
        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin

    def make_debugging_inverter(self, U):

        
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #96I params
                #"mass": 0.00054,
                #"M5": 1.8,
                #"b": 1.5,
                #"c": 0.5,
                #"Ls": 12,
                #"boundary_phases": [1.0, 1.0, 1.0, -1.0],},
        #MDWF_2+1f_64nt128_IWASAKI_b2.25_ls12b+c2_M1.8_ms0.02661_mu0.000678_rhmc_HR_G
                # 64I params
                "mass": 0.0006203,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],},
                #48I params
                # "mass": 0.00078,
                # "M5": 1.8,
                # "b": 1.5,
                # "c": 0.5,
                # "Ls": 24,
                # "boundary_phases": [1.0, 1.0, 1.0, -1.0],},
        )
        
        # l_exact = g.qcd.fermion.zmobius(
        #     #g.convert(U, g.single),
        #     U,
        #     {
        #         "mass": 0.00107,
        #         "M5": 1.8,
        #         "b": 1.0,
        #         "c": 0.0,
        #         "omega": [
        #             1.0903256131299373,
        #             0.9570283702230611,
        #             0.7048886040934104,
        #             0.48979921782791747,
        #             0.328608311201356,
        #             0.21664245377015995,
        #             0.14121112711957107,
        #             0.0907785101745156,
        #             0.05608303440064219 - 0.007537158177840385j,
        #             0.05608303440064219 + 0.007537158177840385j,
        #             0.0365221637144842 - 0.03343945161367745j,
        #             0.0365221637144842 + 0.03343945161367745j,
        #         ],
        #         "boundary_phases": [1.0, 1.0, 1.0, -1.0],
        #     },
        # )
        
        l_sloppy = l_exact.converted(g.single)

        light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-4, maxiter = 10000))
        light_innerH_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-4, maxiter = 200))

        prop_l_sloppy = l_exact.propagator(light_innerH_inverter).grouped(6)
        prop_l_exact = l_exact.propagator(light_innerL_inverter).grouped(6)
        return prop_l_exact, prop_l_sloppy

    ############## make list of complex phases for momentum proj.
    def make_mom_phases(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [-2 * np.pi * np.array(p) / grid.fdimensions for p in self.plist]
       
        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom

    # create Wilson lines from all --> all + dz for all dz in 0,zmax
    def create_WL(self, U):
        W = []
        W.append(g.qcd.gauge.unit(U[2].grid)[0])
        for dz in range(0, self.zmax):
            W.append(g.eval(W[dz] * g.cshift(U[2], 2, dz)))
                
        return W

    '''
    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt(self, prop_f, phases, trafo, tag):

        #g.message("Begin sink smearing")
        #tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        #prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        #g.message("Sink smearing completed")

        corr = g.slice_proton(prop_f, phases, 3) 
        
        if g.rank() == 0:
            save_proton_c2pt_hdf5(corr, tag, my_proton_proj, self.plist)
        del corr 
    '''



    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt_SRC(self, prop_f, phases, trafo, tag):

        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        g.message("Sink smearing completed")

        proton1 = proton_contr(prop_f, prop_f)
        corr = [[g.slice(g.eval(gm*pp),3) for pp in phases] for gm in proton1]
        
        if g.rank() == 0:
            save_proton_c2pt_hdf5(corr, tag, my_gammas, self.plist)
        del corr 

    '''
    def contract_proton_2pt(self,prop_f,phases,trafo):
        proton1 = proton_contr(prop_f, prop_f)
        
        corr = [g.slice(g.eval(proton1*pp),3) for pp in phases]
        
        return corr
    '''

    #function that creates boosted, smeared src.
    def create_src_2pt(self, pos, trafo, grid):
        
        srcD = g.mspincolor(grid)
        g.create.point(srcD, pos)
        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.pos_boost)
        
        return srcDp


class proton_qpdf_measurement(proton_measurement):
   
    def __init__(self, parameters):
        self.zmax = parameters["zmax"]
        self.q = parameters["q"]

        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.plist = [ [0,0, pz, 0] for pz in range(self.pzmin,self.pzmax)]

        self.pol_list = ["P+_Sz+","P+_Sx+","P+_Sx-"]
        #self.Gamma = parameters["gamma"]
        self.t_insert = parameters["t_insert"]
        self.width = parameters["width"]
        self.boost_in = parameters["boost_in"]
        self.boost_out = parameters["boost_out"]
        self.pos_boost = self.boost_in
        self.save_propagators = parameters["save_propagators"]



    def create_fw_prop_QPDF(self, prop_f, W):
        g.message("Creating list of W*prop_f for all z")
        prop_list = [prop_f,]

        for z in range(1,self.zmax):
            prop_list.append(g.eval(W[z]*g.cshift(prop_f,2,z)))
        
        return prop_list  

    def create_bw_seq(self, inverter, prop, trafo):
        tmp_trafo = g.convert(trafo, prop.grid.precision)

        #Make SS propagator
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)

        pp = 2.0 * np.pi * np.array(self.p) / prop.grid.fdimensions
        P = g.exp_ixp(pp)

        # sequential solve through t=insertion_time for all 3 proton polarizations
        src_seq = [g.mspincolor(prop.grid) for i in range(3)]
        dst_seq = []
        #g.mem_report(details=True)
        g.message("starting diquark contractions")
        g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert)
        g.message("diquark contractions done")
        dst_tmp = g.mspincolor(prop.grid)
        for i in range(3):

            dst_tmp @= inverter * g.create.smear.boosted_smearing(tmp_trafo, g.eval(g.gamma[5]* P* g.conj(src_seq[i])), w=self.width, boost=self.boost_out)
            #del src_seq[i]
            dst_seq.append(g.eval(g.gamma[5] * g.conj( dst_tmp)))
        g.message("bw. seq propagator done")
        return dst_seq            


    def contract_QPDF(self, prop_f, prop_bw, phases, tag):
 
        #This and the IO still need work

        for pol in self.pol_list:
            corr = g.slice_trQPDF(prop_bw, prop_f, phases, 3)

            corr_tag = f"{tag}/QPDF/Pol{pol}"
            for z, corr_p in enumerate(corr):
                for i, corr_mu in enumerate(corr_p):
                    p_tag = f"{corr_tag}/pf{self.p}/q{self.q}"
                    for j, corr_t in enumerate(corr_mu):
                        out_tag = f"{p_tag}/{my_gammas[j]}"
                        self.output_correlator.write(out_tag, corr_t)



"""
================================================================================
                Gamma structures and Projection of nucleon states
================================================================================
"""
### Gamma structures
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
#! Add PyQUDA gamma matrices by order
my_pyquda_gammas = [gamma.gamma(15), gamma.gamma(8), gamma.gamma(7), gamma.gamma(1), gamma.gamma(14), gamma.gamma(2), gamma.gamma(13), gamma.gamma(4), gamma.gamma(11), gamma.gamma(0), gamma.gamma(9), gamma.gamma(3), gamma.gamma(5), gamma.gamma(10), gamma.gamma(6), gamma.gamma(12)]
pyq_gamma_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]

### Projection of nucleon states
Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()
CgT5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["T"].tensor() * g.gamma[5].tensor()
CgZ5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["Z"].tensor() * g.gamma[5].tensor()
displaceP = 1 + 0.00000000001
displaceM = 1 - 0.00000000001
Cgplus5 = ( CgT5 * displaceP + 1j * CgZ5 * displaceM ) / np.sqrt(2)
Cgminus5 = ( CgT5 * displaceP - 1j * CgZ5 * displaceM ) / np.sqrt(2)

Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Szm = (g.gamma["I"].tensor() + 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Sxp = (g.gamma["I"].tensor() - 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
Sxm = (g.gamma["I"].tensor() + 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
PpSzp = Pp * Szp
PpSzm = Pp * Szm
PpSxp = Pp * Sxp
PpSxm = Pp * Sxm
#my_projections=["PpSzp", "PpSxp", "PpSxm"]
#my_projections=["PpSzp", "PpSzm", "PpSxp"]
#PolProjections = [PpSzp, PpSxp, PpSxm]
#PolProjections = [PpSzp, PpSzm, PpSxp]
PolProjections = {
    "PpSzp": PpSzp,
    "PpSzm": PpSzm,
    "PpSxp": PpSxp,
    "PpSxm": PpSxm,  
    "PpUnpol": Pp,  
}

#! PyQUDA matrices
epsilon= cp.zeros((3,3,3))
for a in range (3):
    b = (a+1) % 3
    c = (a+2) % 3
    epsilon[a,b,c] = 1
    epsilon[a,c,b] = -1
    
C = gamma.gamma(2) @ gamma.gamma(8)
G5 = gamma.gamma(15)
GZ5 = gamma.gamma(4) @ G5
GT5 = gamma.gamma(8) @ G5

pyquda_gamma_ls = cp.zeros((16, 4, 4), "<c16")
for gamma_idx, gamma_pyq in enumerate(my_pyquda_gammas):
    pyquda_gamma_ls[gamma_idx] = gamma_pyq
    
#! PyQUDA directions for shift
# Xdir = 0
# Ydir = 1
# Zdir = 2
# Tdir = 3
# NXdir = 4
# NYdir = 5
# NZdir = 6
# NTdir = 7


"""
================================================================================
                Used for proton two-point function contraction
================================================================================
"""
ordered_list_of_gammas = [g.gamma[5], g.gamma["T"], g.gamma["T"]*g.gamma[5],
                                      g.gamma["X"], g.gamma["X"]*g.gamma[5], 
                                      g.gamma["Y"], g.gamma["Y"]*g.gamma[5],
                                      g.gamma["Z"], g.gamma["Z"]*g.gamma[5], 
                                      g.gamma["I"], g.gamma["SigmaXT"], 
                                      g.gamma["SigmaXY"], g.gamma["SigmaXZ"], 
                                      g.gamma["SigmaYT"], g.gamma["SigmaYZ"], 
                                      g.gamma["SigmaZT"]
                            ]
def uud_two_point(Q1, Q2, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q2))
    return g(g.color_trace(g.spin_trace(dq) * Q1 + dq * Q1))

def proton_contr(Q1, Q2):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    #Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
    corr = []
    for ig, gm in enumerate(ordered_list_of_gammas):
        Pp = gm
        corr += [g(g.trace(uud_two_point(Q1, Q2, Gamma) * Pp))]
    return corr
    #return g(g.trace(uud_two_point(Q1, Q2, Gamma) * Pp))

"""
================================================================================
                                proton_TMD
================================================================================
"""
class proton_TMD(proton_measurement):

    def __init__(self, parameters):

        self.eta = parameters["eta"] # list of eta
        self.b_z = parameters["b_z"] # largest b_z
        self.b_T = parameters["b_T"] # largest b_T

        self.pf = parameters["pf"] # momentum of final nucleon state; pf = pi + q
        self.plist = parameters["qext"]
        # self.qlist = parameters["qext_PDF"]
        #self.plist = [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in parameters["qext"] for y in parameters["qext"] for z in [0]}]
        #self.plist = [[x,y,z,0] for x in parameters["qext"] for y in parameters["qext"] for z in parameters["qext"]] # generating momentum transfers for TMD
        #self.qlist = [[x,y,z,0] for x in parameters["qext_PDF"] for y in parameters["qext_PDF"] for z in parameters["qext_PDF"]] # generating momentum transfers for PDF
        #self.pilist = [[parameters["pf"][0]-x,parameters["pf"][1]-y,parameters["pf"][2]-z,0] for x in parameters["qext"] for y in parameters["qext"] for z in parameters["qext"]] # generating pi = pf - q
        self.pilist = parameters["p_2pt"]  # 2pt momentum

        self.width = parameters["width"] # Gaussian smearing width
        self.boost_in = parameters["boost_in"] # ?? Forward propagator boost smearing
        self.boost_out = parameters["boost_out"] # ?? Backward propagator boost smearing
        self.pos_boost = self.boost_in # Forward propagator boost smearing for 2pt

        self.pol_list = parameters["pol"] # projection of nucleon state
        self.t_insert = parameters["t_insert"] # time separation of three point function

        self.save_propagators = parameters["save_propagators"] # if save propagators
    
    ############## make list of complex phases for momentum proj.
    def make_mom_phases_2pt(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [-2 * np.pi * np.array(pi) / grid.fdimensions for pi in self.pilist] # pilist is the pf-q

        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom
    def make_mom_phases_3pt(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [2 * np.pi * np.array(p) / grid.fdimensions for p in self.plist] # plist is the q for TMD

        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom
    def make_mom_phases_PDF(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [2 * np.pi * np.array(p) / grid.fdimensions for p in self.qlist] # qlist is the q for PDF

        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom
    
    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt_TMD(self, prop_f, phases, trafo, tag, interpolation = "5"):

        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        g.message("Sink smearing completed")

        #TODO: Jinchen, new interpolation operator
        if interpolation == "5":
            dq = g.qcd.baryon.diquark(g(prop_f * Cg5), g(Cg5 * prop_f))
        elif interpolation == "T5":
            dq = g.qcd.baryon.diquark(g(prop_f * CgT5), g(CgT5 * prop_f)) 
        elif interpolation == "Z5":
            dq = g.qcd.baryon.diquark(g(prop_f * CgZ5), g(CgZ5 * prop_f)) 
        elif interpolation == "p5":
            dq = g.qcd.baryon.diquark(g(prop_f * Cgminus5), g(Cgplus5 * prop_f))
            # dq = g.qcd.baryon.diquark(g(prop_f * Cgminus5), g(Cgminus5 * prop_f))
        else:
            raise ValueError("Invalid interpolation operator")
        
        proton1 = g(g.spin_trace(dq) * prop_f + dq * prop_f)
        prop_unit = g.mspincolor(prop_f.grid)
        prop_unit = g.identity(prop_unit)
        corr = g.slice_trDA([prop_unit], [proton1], phases,3)
        corr = [[corr[0][i][j] for i in range(0, len(corr[0]))] for j in range(0, len(corr[0][0])) ]

        if g.rank() == 0:
            save_proton_c2pt_hdf5(corr, tag, my_gammas, self.pilist)
        del corr
        
    #! PyQUDA: contract 2pt TMD
    def contract_2pt_TMD_pyquda(self, prop_f, phases, trafo, tag, interpolation = "5"): 
        if interpolation == "5":
            interp_opt = C @ G5
        elif interpolation == "T5":
            interp_opt = C @ GT5
        elif interpolation == "Z5":
            interp_opt = C @ GZ5
        else:
            raise ValueError("Invalid interpolation operator")
        
        
        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        g.message("Sink smearing completed")
        
        prop_f_pyq = gpt.LatticePropagatorGPT(prop_f, GEN_SIMD_WIDTH)
        
        Lt = np.shape(prop_f_pyq.data)[1]
        
        P_2pt_gamma = cp.zeros((16, Lt, 4, 4), "<c16")
        for gamma_idx, gamma_pyq in enumerate(my_pyquda_gammas):
            P_2pt = cp.zeros((Lt, 4, 4), "<c16")
            P_2pt[:] = gamma_pyq
            P_2pt_gamma[gamma_idx] = P_2pt
        
        corr = (
                - contract(
                "abc, def, pwtzyx, ij, kl, gtmn, wtzyxikad, wtzyxjlbe, wtzyxmncf->gpt",
                epsilon,    epsilon,    phases,    interp_opt,    interp_opt,    P_2pt_gamma,
                prop_f_pyq.data,  prop_f_pyq.data,  prop_f_pyq.data,
                ) 
                - contract(
                    "abc, def, pwtzyx, ij, kl, gtmn, wtzyxikad, wtzyxjnbe, wtzyxmlcf->gpt",
                    epsilon,    epsilon,    phases,    interp_opt,    interp_opt,    P_2pt_gamma,
                    prop_f_pyq.data,  prop_f_pyq.data,  prop_f_pyq.data,
                )
            )
        corr_collect = core.gatherLattice(corr.get(), [2, -1, -1, -1])
        
        if g.rank() == 0:
            save_proton_c2pt_hdf5(corr_collect, tag, my_gammas, self.pilist)
        del corr, corr_collect

    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt_TMD_old(self, prop_f, phases, trafo, tag):

        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        g.message("Sink smearing completed")

        proton1 = proton_contr(prop_f, prop_f)
        corr = [[g.slice(g.eval(gm*pp),3) for pp in phases] for gm in proton1]
        
        if g.rank() == 0:
            save_proton_c2pt_hdf5(corr, tag, my_gammas, self.pilist)
        del corr 

    def create_fw_prop_TMD(self, prop_f, W, W_index_list):
        g.message("Creating list of W*prop_f with shift bT and 2*bz")
        prop_list = []
        
        for i, idx in enumerate(W_index_list):

            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            #prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[i] * g.cshift(g.cshift(prop_f,transverse_direction,current_b_T),2,round(2*current_bz)))*g.gamma[5])))
            prop_list.append(g.eval(W[i] * g.cshift(g.cshift(prop_f,transverse_direction,current_b_T),2,round(2*current_bz)))) 
        return prop_list
    
    #! PyQUDA: create forward propagator for CG TMD, support +- shift
    def create_fw_prop_TMD_CG_pyquda(self, prop_f_pyq, W_index, WL_indices_previous):
        current_b_T = W_index[0]
        current_bz = W_index[1]
        transverse_direction = W_index[3] # 0, 1
        Zdir = 2
        
        previous_b_T = WL_indices_previous[0]
        previous_bz = WL_indices_previous[1]
        
        prop_shift_pyq = prop_f_pyq.shift(round(current_b_T - previous_b_T), transverse_direction).shift(round(current_bz - previous_bz), Zdir)

        return prop_shift_pyq

    def create_fw_prop_TMD_CG(self, prop_f, W_index_list):
        g.message("Creating list of prop_f with shift bT and bz")
        prop_list = []
        
        for i, idx in enumerate(W_index_list):

            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]

            prop_list.append(g.eval(g.cshift(g.cshift(prop_f,transverse_direction,current_b_T),2,round(current_bz)))) 

        return prop_list

    #! PyQUDA: create forward propagator for CG TMD, support +- shift
    def create_fw_prop_PDF_GI_pyquda(self, gauge, prop_f_pyq, W_index, WL_indices_previous):

        current_bz = W_index[1]
        previous_bz = WL_indices_previous[1]

        #! PyQUDA: forward prop
        for spin in range(4):
            for color in range(3):
                fermion = prop_f_pyq.getFermion(spin, color)
                if current_bz - previous_bz == 0:
                    fermion_shift = fermion
                elif current_bz - previous_bz == 1:
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 2)
                elif current_bz - previous_bz == -1:
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 6) # -z direction
                else:
                    raise ValueError("Invalid shift for PDF Wilson line")
                #\psi'(x)=U_\mu(x)\psi(x+\hat\mu)0,1,2,3 for x,y,z,t; 4,5,6,7 for -x,-y,-z,-t
                prop_f_pyq.setFermion(fermion_shift, spin, color)

        return prop_f_pyq
    
    def create_fw_prop_PDF(self, prop_f, W, W_index_list):
        g.message("Creating list of W*prop_f")
        prop_list = []
        
        for i, idx in enumerate(W_index_list):

            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            assert current_b_T == 0
            assert current_eta == 0
            assert transverse_direction == 0

            prop_list.append(g.eval(W[i] * g.cshift(g.cshift(prop_f,0,0),2,round(current_bz)))) 
        return prop_list

    def create_bw_seq_Pyquda(self, dirac, prop, trafo, flavor, origin=None, interpolation = "5"):
        tmp_trafo = g.convert(trafo, prop.grid.precision) #Need later for mixed precision solver
        
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)
        
        pp = 2.0 * np.pi * np.array(self.pf) / prop.grid.fdimensions
        P = g.exp_ixp(pp, origin)
        
        src_seq = [g.mspincolor(prop.grid) for i in range(len(self.pol_list))]
        dst_seq = []
        dst_tmp = g.mspincolor(prop.grid)
        
        #g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert, flavor)
        for i, pol in enumerate(self.pol_list):

            if (flavor == 1): 
                g.message("starting diquark contractions for up quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgZ5, PolProjections[pol]) 
                else:
                    raise ValueError("Invalid interpolation operator")
                
            elif (flavor == 2):
                g.message("starting diquark contractions for down quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.down_quark_insertion(prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.down_quark_insertion(prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.down_quark_insertion(prop, CgZ5, PolProjections[pol])     
                else:
                    raise ValueError("Invalid interpolation operator")
            else: 
                raise Exception("Unknown flavor for backward sequential src construction")
        
            # sequential solve through t=t_insert
            src_seq_t = g.lattice(src_seq[i])
            src_seq_t[:] = 0
            src_seq_t[:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]] = src_seq[i][:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]]

            g.message("diquark contractions for Polarization ", i, pol, " done")
        
            smearing_input = g.eval(g.gamma[5]*P*g.adj(src_seq_t))

            tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input,w=self.width, boost=self.boost_out)

            src_pyquda = gpt.LatticePropagatorGPT(tmp_prop, GEN_SIMD_WIDTH)
            prop_pyquda = core.invertPropagator(dirac, src_pyquda, 1, 0) # NOTE or "prop_pyquda = core.invertPropagator(dirac, src_pyquda, 0)" depends on the quda version
            dst_tmp = g.mspincolor(prop.grid)
            gpt.LatticePropagatorGPT(dst_tmp, GEN_SIMD_WIDTH, prop_pyquda)
            del src_pyquda, prop_pyquda

            dst_seq.append(g.eval(g.adj(dst_tmp) * g.gamma[5]))

        return dst_seq
    
    #! PyQUDA: get backward propagator through sequential source for U and D
    def create_bw_seq_Pyquda_pyquda(self, dirac, prop, trafo, flavor, origin=None, interpolation = "5"):
        tmp_trafo = g.convert(trafo, prop.grid.precision) #Need later for mixed precision solver
        
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)
        
        pp = 2.0 * np.pi * np.array(self.pf) / prop.grid.fdimensions
        P = g.exp_ixp(pp, origin)
        
        src_seq = [g.mspincolor(prop.grid) for i in range(len(self.pol_list))]
        dst_seq = []
        
        for i, pol in enumerate(self.pol_list):

            if (flavor == 1): 
                g.message("starting diquark contractions for up quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgZ5, PolProjections[pol])
                elif interpolation == "p5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, Cgminus5, PolProjections[pol])
                else:
                    raise ValueError("Invalid interpolation operator")
                
            elif (flavor == 2):
                g.message("starting diquark contractions for down quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.down_quark_insertion(prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.down_quark_insertion(prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.down_quark_insertion(prop, CgZ5, PolProjections[pol])
                elif interpolation == "p5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, Cgminus5, PolProjections[pol])
                else:
                    raise ValueError("Invalid interpolation operator")
            else: 
                raise Exception("Unknown flavor for backward sequential src construction")
        
            # sequential solve through t=t_insert
            src_seq_t = g.lattice(src_seq[i])
            src_seq_t[:] = 0
            src_seq_t[:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]] = src_seq[i][:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]]

            g.message("diquark contractions for Polarization ", i, pol, " done")
        
            smearing_input = g.eval(g.gamma[5]*P*g.adj(src_seq_t))

            tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input,w=self.width, boost=self.boost_out)

            src_pyquda = gpt.LatticePropagatorGPT(tmp_prop, GEN_SIMD_WIDTH)
            prop_pyquda = core.invertPropagator(dirac, src_pyquda, 1, 0) # NOTE or "prop_pyquda = core.invertPropagator(dirac, src_pyquda, 0)" depends on the quda version
            
            prop_pyquda_contracted = contract( "wtzyxijfc, ik -> wtzyxjkcf", prop_pyquda.data.conj(), G5 )
            del src_pyquda, prop_pyquda
            
            dst_seq.append(prop_pyquda_contracted)
            
        dst_seq = cp.asarray(dst_seq)

        return dst_seq

    def create_bw_seq(self, inverter, prop, trafo, flavor, origin=None, interpolation = "5"):
        tmp_trafo = g.convert(trafo, prop.grid.precision) #Need later for mixed precision solver
        
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)
        
        pp = 2.0 * np.pi * np.array(self.pf) / prop.grid.fdimensions
        P = g.exp_ixp(pp, origin)
        
        src_seq = [g.mspincolor(prop.grid) for i in range(len(self.pol_list))]
        dst_seq = []
        dst_tmp = g.mspincolor(prop.grid)
        
        #g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert, flavor)
        for i, pol in enumerate(self.pol_list):

            if (flavor == 1): 
                g.message("starting diquark contractions for up quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgZ5, PolProjections[pol]) 
                else:
                    raise ValueError("Invalid interpolation operator")
                
            elif (flavor == 2):
                g.message("starting diquark contractions for down quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.down_quark_insertion(prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.down_quark_insertion(prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.down_quark_insertion(prop, CgZ5, PolProjections[pol]) 
                else:
                    raise ValueError("Invalid interpolation operator")
                
            else: 
                raise Exception("Unknown flavor for backward sequential src construction")
        
            # sequential solve through t=t_insert
            src_seq_t = g.lattice(src_seq[i])
            src_seq_t[:] = 0
            src_seq_t[:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]] = src_seq[i][:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]]

            g.message("diquark contractions for Polarization ", i, " done")
        
            # FIXME smearing_input = g.eval(g.gamma[5]*P*g.conj(src_seq_t))
            smearing_input = g.eval(g.gamma[5]*P*g.adj(src_seq_t))

            tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input,w=self.width, boost=self.boost_out)

            dst_tmp = g.eval(inverter * tmp_prop)           
            # FIXME dst_seq.append(g.eval(g.gamma[5] * g.conj(dst_tmp)))
            dst_seq.append(g.eval(g.adj(dst_tmp) * g.gamma[5]))

        g.message("bw. seq propagator done")
        return dst_seq
    
    def contract_TMD(self, prop_f, prop_bw_seq, phases, W_index, tag, iW):
        
        corr = g.slice_trDA(prop_bw_seq, prop_f, phases,3)

        for pol_index in range(len(prop_bw_seq)):
            pol_tag = tag + "." + self.pol_list[pol_index]
            
            corr_write = [corr[pol_index]]  
            
            if g.rank() == pol_index:
                #print('g.rank():',g.rank(), ', pol_tag:', pol_tag)
                save_qTMD_proton_hdf5_subset(corr_write, pol_tag, my_gammas, self.plist, [W_index], iW, self.t_insert)

    def contract_PDF(self, prop_f, prop_bw_seq, phases, W_index, tag, iW):
        
        corr = g.slice_trDA(prop_bw_seq, prop_f, phases,3)

        for pol_index in range(len(prop_bw_seq)):
            pol_tag = tag + "." + self.pol_list[pol_index]
            
            corr_write = [corr[pol_index]]  
            
            if g.rank() == pol_index:
                #print('g.rank():',g.rank(), ', pol_tag:', pol_tag)
                save_qTMD_proton_hdf5_subset(corr_write, pol_tag, my_gammas, self.qlist, [W_index], iW, self.t_insert)
    
    def create_PDF_Wilsonline_index_list(self):
        index_list = []
        
        for current_bz in range(0, self.b_z + 1):
            # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
            index_list.append([0, current_bz, 0, 0])
            
        for current_bz in range(0, self.b_z + 1):
            # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
            if current_bz != 0:
                index_list.append([0, -current_bz, 0, 0])
                    
        return index_list
    
    def create_TMD_Wilsonline_index_list(self):
        index_list = []
        
        for transverse_direction in [0,1]:
            for current_eta in self.eta:
                
                if current_eta <= 12:
                    for current_bz in range(0, min([self.b_z+1, current_eta+1])):
                        for current_b_T in range(0, min([self.b_T+1, current_eta+1])):
                            
                            # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
                            index_list.append([current_b_T, current_bz, current_eta, transverse_direction])
                            
                            # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
                            index_list.append([current_b_T, -current_bz, -current_eta, transverse_direction])
                else:
                    # create Wilson lines from all to all + (eta+0) + b_perp - (eta-0)
                    for current_b_T in range(0, min([self.b_T+1, current_eta+1])):
                        index_list.append([current_b_T, 0, current_eta, transverse_direction])
                    
        return index_list
        
    def create_TMD_Wilsonline_index_list_CG(self, grid):
        index_list = []
        
        for transverse_direction in [0,1]:
            for current_bz in range(0, self.b_z+1):
                for current_b_T in range(0, self.b_T+1):
            
                    # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
                    index_list.append([current_b_T, current_bz, 0, transverse_direction])
                    
                    # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
                    #if current_bz != 0:
                    #    index_list.append([current_b_T, -current_bz, 0, transverse_direction])
                    
        return index_list
    
    #! PyQUDA: create Wilson line index list for CG TMD
    def create_TMD_Wilsonline_index_list_CG_pyquda(self):
        index_list_trans0 = []
        index_list_trans1 = []
        
        for current_bz in range(0, self.b_z+1):
            for current_b_T in range(0, self.b_T+1):
                # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
                index_list_trans0.append([current_b_T, current_bz, 0, 0])
                index_list_trans1.append([current_b_T, current_bz, 0, 1])

                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, 0, 0])
                    index_list_trans1.append([current_b_T, -current_bz, 0, 1])

                #if current_b_T != 0:
                #    index_list_trans0.append([-current_b_T, current_bz, 0, 0])
                #    index_list_trans1.append([-current_b_T, current_bz, 0, 1])
                #    if current_bz != 0:
                #        index_list_trans0.append([-current_b_T, -current_bz, 0, 0])
                #        index_list_trans1.append([-current_b_T, -current_bz, 0, 1])
                
        # Reorder index lists to minimize differences between adjacent indices
        def reorder_indices(index_list):
            # Sort by bT first, then bz to minimize jumps
            sorted_list = sorted(index_list, key=lambda x: (x[0], x[1]))
            reordered = []
            
            # Process pairs of indices to minimize differences
            i = 0
            while i < len(sorted_list)-1:
                curr = sorted_list[i]
                next = sorted_list[i+1]
                
                # If difference is more than 1 in either bT or bz, try to find better match
                if abs(curr[0] - next[0]) > 1 or abs(curr[1] - next[1]) > 1:
                    # Look ahead for better match
                    best_match = next
                    best_diff = max(abs(curr[0] - next[0]), abs(curr[1] - next[1]))
                    
                    for j in range(i+2, len(sorted_list)):
                        candidate = sorted_list[j]
                        diff = max(abs(curr[0] - candidate[0]), abs(curr[1] - candidate[1]))
                        if diff < best_diff:
                            best_match = candidate
                            best_diff = diff
                    
                    # Swap to get better ordering
                    if best_match != next:
                        idx = sorted_list.index(best_match)
                        sorted_list[i+1], sorted_list[idx] = sorted_list[idx], sorted_list[i+1]
                
                reordered.append(curr)
                i += 1
                
            if i < len(sorted_list):
                reordered.append(sorted_list[-1])
                
            return reordered
            
        index_list_trans0 = reorder_indices(index_list_trans0)
        index_list_trans1 = reorder_indices(index_list_trans1)
                
        return index_list_trans0, index_list_trans1
    
    def create_PDF_Wilsonline(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]
        assert bt_index == 0
        assert eta_index == 0
        assert transverse_dir == 0
        
        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
        WL = prv_link

        if bz_index >= 0:
            for dz in range(0, bz_index):
                WL = g.eval(prv_link * g.cshift(U[2], 2, dz))
                prv_link = WL
        else:
            for dz in range(0, abs(bz_index)):
                WL = g.eval(prv_link * g.adj(g.cshift(U[2],2, -dz-1)))
                prv_link = WL

        return WL
    
    def create_TMD_Wilsonline(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]
        
        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
        WL = prv_link

        if eta_index+bz_index >= 0:
            for dz in range(0, eta_index+bz_index):
                WL = g.eval(prv_link * g.cshift(U[2], 2, dz))
                prv_link = WL
        else:
            for dz in range(0, abs(eta_index+bz_index)):
                WL = g.eval(prv_link * g.adj(g.cshift(U[2],2, -dz-1)))
                prv_link = WL
        
        # dx and bt_index are >=0
        for dx in range(0, bt_index):
            WL=g.eval(prv_link * g.cshift(g.cshift(U[transverse_dir], 2, eta_index+bz_index),transverse_dir, dx))
            prv_link=WL

        if eta_index-bz_index >= 0:
            for dz in range(0, eta_index-bz_index):
                WL=g.eval(prv_link * g.adj(g.cshift(g.cshift(g.cshift(U[2], 2, eta_index+bz_index-1), transverse_dir, bt_index),2,-dz)))
                prv_link=WL
        else:
            for dz in range(0, abs(eta_index-bz_index)):
                WL=g.eval(prv_link * g.cshift(g.cshift(g.cshift(U[2], 2, eta_index+bz_index), transverse_dir, bt_index),2,dz))
                prv_link=WL

        return WL

    def create_TMD_Wilsonline_CG(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]

        return g.qcd.gauge.unit(U[2].grid)[0]
            
    def create_TMD_Wilsonline_CG_Tlink(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]
        
        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
        WL = prv_link
        
        # dx and bt_index are >=0
        for dx in range(0, bt_index):
            WL=g.eval(prv_link * g.cshift(g.cshift(U[transverse_dir], 2, eta_index+bz_index),transverse_dir, dx))
            prv_link=WL

        return WL
    
    def down_quark_insertion(self, Q, Gamma, P):
        #eps_abc eps_a'b'c'Gamma_{beta alpha}Gamma_{beta'alpha'}P_{gamma gamma'}
        # * ( Q^beta'beta_b'b Q^gamma'gamma_{c'c} -  Q^beta'gamma_b'c Q^gamma'beta_{c'b} )
        
        eps = g.epsilon(Q.otype.shape[2])
        
        R = g.lattice(Q)
        
        PDu = g(g.spin_trace(P*Q))

        GtDG = g.eval(g.transpose(Gamma)*Q*Gamma)

        GtDG = g.separate_color(GtDG)
        PDu = g.separate_color(PDu)
        
        GtD = g.eval(g.transpose(Gamma)*Q)
        PDG = g.eval(P*Q*Gamma)
        
        GtD = g.separate_color(GtD)
        PDG = g.separate_color(PDG)
        
        D = {x: g.lattice(GtDG[x]) for x in GtDG}

        for d in D:
            D[d][:] = 0
            
        for i1, sign1 in eps:
            for i2, sign2 in eps:
                D[i1[0], i2[0]] += -sign1 * sign2 * g.transpose((PDu[i2[2], i1[2]] * GtDG[i2[1], i1[1]] - GtD[i2[1],i1[2]] * PDG[i2[2], i1[1]]))
                
        g.merge_color(R, D)
        return R

    #Qlua definition, reproduce the results as Chroma difinition
    def up_quark_insertion(self, Qu, Qd, Gamma, P):

        eps = g.epsilon(Qu.otype.shape[2])
        R = g.lattice(Qu)

        Du_sep = g.separate_color(Qu)
        GDd = g.eval(Gamma * Qd)
        GDd = g.separate_color(GDd)

        PDu = g.eval(P*Qu)
        PDu = g.separate_color(PDu)

        # ut
        DuP = g.eval(Qu * P)
        DuP = g.separate_color(DuP)
        TrDuP = g(g.spin_trace(Qu * P))
        TrDuP = g.separate_color(TrDuP)
        
        # s2ds1b
        GtDG = g.eval(g.transpose(Gamma)*Qd*Gamma)
        GtDG = g.separate_color(GtDG)

        #sum color indices
        D = {x: g.lattice(GDd[x]) for x in GDd}
        for d in D:
            D[d][:] = 0

        for i1, sign1 in eps:
            for i2, sign2 in eps:
                D[i2[2], i1[2]] += -sign1 * sign2 * (P * g.spin_trace(GtDG[i1[1],i2[1]]*g.transpose(Du_sep[i1[0],i2[0]]))
                                    + g.transpose(TrDuP[i1[0],i2[0]] * GtDG[i1[1],i2[1]])
                                    + PDu[i1[0],i2[0]] * g.transpose(GtDG[i1[1],i2[1]])
                                    + g.transpose(GtDG[i1[0],i2[0]]) * DuP[i1[1],i2[1]])
        
        g.merge_color(R, D)

        return R

    # Chroma definition, reproduce the results as Qlua definition
    '''
    def up_quark_insertion(Qu, Qd, Gamma, P):

        eps = g.epsilon(Qu.otype.shape[2])
        R = g.lattice(Qu)
        Dut = g.lattice(Qu)

        Du_sep = g.separate_color(Qu)
        GDd = g.eval(Gamma * Qd)
        GDd = g.separate_color(GDd)

        #first term & second term
        GDd = g.eval(Gamma * Qd)
        GDd = g.separate_color(GDd)

        DuG = g.eval(Qu * Gamma)
        DuG = g.separate_color(DuG)

        #third term
        Du_sep = g.separate_color(Qu)
        Du_spintransposed = {x: g.lattice(Du_sep[x]) for x in Du_sep}
        for d in Du_spintransposed:
            Du_spintransposed[d] = g(g.transpose(Du_sep[d]))
        g.merge_color(Dut,Du_spintransposed)

        PDut = g.eval(g.transpose(P) * Dut)
        PDut = g.separate_color(PDut)
        GDuG = g.eval(Gamma * Qu * Gamma)
        GDuG = g.separate_color(GDuG)    

        #fourth term
        #GDuG = g.eval(Gamma * Qu * Gamma)
        #GDuG = g.separate_color(GDuG)
        DuP_trace = g(g.spin_trace(Qu * P))
        DuP_trace = g.separate_color(DuP_trace)

        #sum color indices
        D = {x: g.lattice(GDd[x]) for x in GDd}
        for d in D:
            D[d][:] = 0

        for i1, sign1 in eps:
            for i2, sign2 in eps:
                tmp = -sign1 * sign2 * (GDd[i1[1],i2[1]] * g.transpose(DuG[i1[0],i2[0]]) * g.transpose(P)
                                    + g.spin_trace(GDd[i1[1],i2[1]] * g.transpose(DuG[i1[0],i2[0]])) * g.transpose(P)
                                    - PDut[i1[1],i2[1]] * GDuG[i1[0],i2[0]]
                                    - DuP_trace[i1[0],i2[0]] * GDuG[i1[1],i2[1]])
                D[i2[2], i1[2]] += g.transpose(tmp)
        
        g.merge_color(R, D)
        return R
    '''
