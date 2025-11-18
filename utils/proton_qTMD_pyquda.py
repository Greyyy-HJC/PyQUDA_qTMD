from pyquda import init, LatticeInfo
from pyquda_utils import core, gamma
from utils.boosted_smearing_pyquda import boosted_smearing
from utils.io_corr import save_proton_c2pt_hdf5
from utils.tools import _get_xp_from_array, _ensure_backend, mpi_print


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
    def contract_2pt_TMD(self, latt_info, prop_f, phases, trafo, tag, interpolator = "5"): 
        if interpolator == "5":
            gamma_insert = Cg5
        elif interpolator == "T5":
            gamma_insert = CgT5
        elif interpolator == "Z5":
            gamma_insert = CgZ5
        else:
            raise ValueError(f"Invalid interpolator: {interpolator}")
        
        
        mpi_print(latt_info, "Begin sink smearing")
        prop_f = boosted_smearing(trafo, prop_f, w=self.width, boost=self.pos_boost)
        mpi_print(latt_info, "Sink smearing completed")
        
        xp = _get_xp_from_array(prop_f.data)
        P_2pt_gamma = xp.zeros((16, latt_info.Lt, 4, 4), "<c16")
        for gamma_idx, gamma_pyq in enumerate(my_pyquda_gammas):
            P_2pt = xp.zeros((latt_info.Lt, 4, 4), "<c16")
            P_2pt[:] = gamma_pyq
            P_2pt_gamma[gamma_idx] = P_2pt
            
        epsilon= xp.zeros((3,3,3))
        for a in range (3):
            b = (a+1) % 3
            c = (a+2) % 3
            epsilon[a,b,c] = 1
            epsilon[a,c,b] = -1
        
        corr = (
                - xp.einsum(
                "abc, def, pwtzyx, ij, kl, gtmn, wtzyxikad, wtzyxjlbe, wtzyxmncf->gpt",
                epsilon,    epsilon,    phases,    gamma_insert,    gamma_insert,    P_2pt_gamma,
                prop_f.data,  prop_f.data,  prop_f.data,
                ) 
                - xp.einsum(
                    "abc, def, pwtzyx, ij, kl, gtmn, wtzyxikad, wtzyxjnbe, wtzyxmlcf->gpt",
                    epsilon,    epsilon,    phases,    gamma_insert,    gamma_insert,    P_2pt_gamma,
                    prop_f.data,  prop_f.data,  prop_f.data,
                )
            )
        corr_collect = core.gatherLattice(corr.get(), [2, -1, -1, -1])
        
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