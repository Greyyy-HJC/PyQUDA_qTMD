from pyquda_utils import core, gamma
from utils.boosted_smearing_pyquda import boosted_smearing
from utils.io_corr import save_proton_c2pt_hdf5
from utils.tools import _get_xp_from_array, mpi_print, _asarray_on_queue


my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
#! Add PyQUDA gamma matrices by order
my_pyquda_gammas = [gamma.gamma(15), gamma.gamma(8), gamma.gamma(7), gamma.gamma(1), gamma.gamma(14), gamma.gamma(2), gamma.gamma(13), gamma.gamma(4), gamma.gamma(11), gamma.gamma(0), gamma.gamma(9), gamma.gamma(3), gamma.gamma(5), gamma.gamma(10), gamma.gamma(6), gamma.gamma(12)]
pyquda_gammas_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]

Cg5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(15)
CgT5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(7)
CgZ5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(11)


"""
================================================================================
                                proton_TMD
================================================================================
"""
class proton_TMD():
    def __init__(self, parameters):

        self.eta = parameters["eta"] # list of eta
        self.b_z = parameters["b_z"] # largest b_z
        self.b_T = parameters["b_T"] # largest b_T

        self.pf = parameters["pf"] # momentum of final nucleon state; pf = pi + q
        self.plist = parameters["qext"]
        self.qlist = parameters["qext_PDF"]
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
        
    #! PyQUDA: contract 2pt TMD
    def contract_2pt_TMD(self, latt_info, prop_f, phases, tag, interpolator = "5"): 
        if interpolator == "5":
            gamma_insert = Cg5
        elif interpolator == "T5":
            gamma_insert = CgT5
        elif interpolator == "Z5":
            gamma_insert = CgZ5
        else:
            raise ValueError(f"Invalid interpolator: {interpolator}")
        
        
        mpi_print(latt_info, "Begin sink smearing")
        prop_f = boosted_smearing(prop_f, w=self.width, boost=self.pos_boost)
        mpi_print(latt_info, "Sink smearing completed")
        
        xp = _get_xp_from_array(prop_f.data)
        P_2pt_gamma_host = xp.zeros((16, latt_info.Lt, 4, 4))
        P_2pt_gamma = _asarray_on_queue(P_2pt_gamma_host, xp, prop_f.data)

        for gamma_idx, gamma_pyq_host in enumerate(my_pyquda_gammas):
            gamma_device = _asarray_on_queue(gamma_pyq_host, xp, prop_f.data)
            
            P_2pt_local = _asarray_on_queue(xp.zeros((latt_info.Lt, 4, 4)), xp, prop_f.data)
            P_2pt_local[:] = gamma_device
            P_2pt_gamma[gamma_idx] = P_2pt_local
            
        epsilon_host = xp.zeros((3,3,3))
        for a in range (3):
            b = (a+1) % 3
            c = (a+2) % 3
            epsilon_host[a,b,c] = 1
            epsilon_host[a,c,b] = -1
        epsilon = _asarray_on_queue(epsilon_host, xp, prop_f.data)
        
        phases = _asarray_on_queue(phases, xp, prop_f.data)
        gamma_insert = _asarray_on_queue(gamma_insert, xp, prop_f.data)
        
        
        #! Optimized version of the 2pt TMD contraction
        # --- Term 1 ---
        # original equation: -1 * [epsilon(abc)*epsilon(def)*G(ij)*G(kl)*P2pt(gtmn)*P1(ikad)*P2(jlbe)*P3(mncf)*Phases]
        # structure: Prop1 and Prop2 are directly connected to Sink(ij) and Src(kl); Prop3 is self-closed(mn)

        # 1. [Sink Block] contract Sink color(abc), spin(ij) and Prop1, Prop2
        #    Indices: P1(i,k,a,d), P2(j,l,b,e), E(a,b,c), G(i,j) -> Result(k,l,d,e,c)
        #    Note: c is the sink color of Prop3
        term1_sink = xp.einsum(
            "abc, ij, wtzyxikad, wtzyxjlbe -> wtzyxklcde",
            epsilon, gamma_insert, prop_f.data, prop_f.data,
            optimize=True
        )

        # 2. [P3 Block] Prop3 is self-closed with P_2pt
        #    Indices: P3(m,n,c,f), P2pt(g,t,m,n) -> Result(g,c,f)
        #    Note: here m,n are contracted
        term1_p3 = xp.einsum(
            "gtmn, wtzyxmncf -> gwtzyxcf",
            P_2pt_gamma, prop_f.data,
            optimize=True
        )

        # 3. [Final Assembly] combine two parts
        #    Indices: Sink(k,l,d,e,c), P3(g,c,f), E(d,e,f), G(k,l), Phases
        #    Remaining: g, p, t
        term1 = xp.einsum(
            "def, pwtzyx, kl, wtzyxklcde, gwtzyxcf -> gpt",
            epsilon, phases, gamma_insert, term1_sink, term1_p3,
            optimize=True
        )

        # clean up memory
        del term1_sink, term1_p3


        # --- Term 2 ---
        # original equation: -1 * [epsilon(abc)*epsilon(def)*G(ij)*G(kl)*P2pt(gtmn)*P1(ikad)*P2(jnbe)*P3(mlcf)*Phases]
        # structure: Prop1(k) is connected to Prop3(l) in Src; Prop2(n) is connected to Prop3(m) through P2pt... This is a big loop.

        # 1. [Sink Block] contract Sink color(abc), spin(ij) and Prop1, Prop2
        #    Indices: P1(i,k,a,d), P2(j,n,b,e), E(a,b,c), G(i,j) -> Result(k,n,d,e,c)
        #    Note: l is changed to n
        term2_sink = xp.einsum(
            "abc, ij, wtzyxikad, wtzyxjnbe -> wtzyxkncde",
            epsilon, gamma_insert, prop_f.data, prop_f.data,
            optimize=True
        )

        # 2. [P3 Block] Prop3 is connected to P2pt
        #    Indices: P3(m,l,c,f), P2pt(g,t,m,n) -> Result(g,n,l,c,f)
        #    Note: m is contracted, but n (connected to Prop2) and l (connected to Gamma) are kept
        term2_p3 = xp.einsum(
            "gtmn, wtzyxmlcf -> gwtzyxnlcf",
            P_2pt_gamma, prop_f.data,
            optimize=True
        )

        # 3. [Final Assembly]
        #    Indices: 
        #      Sink: k, n, d, e, c
        #      P3:   g, n, l, c, f
        #      Gamma: k, l
        #      Eps:   d, e, f
        #    contracted path: 
        #      k (Sink-Gamma), l (P3-Gamma), n (Sink-P3), c (Sink-P3), def (Eps-Sink-P3)
        term2 = xp.einsum(
            "def, pwtzyx, kl, wtzyxkncde, gwtzyxnlcf -> gpt",
            epsilon, phases, gamma_insert, term2_sink, term2_p3,
            optimize=True
        )

        # clean up memory
        del term2_sink, term2_p3

        # --- Final Result ---
        # original code is (- Einsum1 - Einsum2)
        corr = - term1 - term2

        corr_collect = core.gatherLattice(xp.asnumpy(corr), [2, -1, -1, -1])
        
        
        if latt_info.mpi_rank == 0:
            save_proton_c2pt_hdf5(corr_collect, tag, my_gammas, self.pilist)
        del corr, corr_collect
    
        
    def create_TMD_Wilsonline_index_list_CG(self):
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
                    
    #! PyQUDA: create forward propagator for CG TMD, support +- shift
    def create_fw_prop_TMD_CG(self, prop_f, W_index, WL_indices_previous):
        current_b_T = W_index[0]
        current_bz = W_index[1]
        transverse_direction = W_index[3] # 0, 1
        Zdir = 2
        
        previous_b_T = WL_indices_previous[0]
        previous_bz = WL_indices_previous[1]
        
        prop_shift = prop_f.shift(round(current_b_T - previous_b_T), transverse_direction).shift(round(current_bz - previous_bz), Zdir)

        return prop_shift
    
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
    
    #! PyQUDA: create forward propagator for CG TMD, support +- shift
    def create_fw_prop_PDF_GI(self, gauge, prop_f_pyq, W_index, WL_indices_previous):

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