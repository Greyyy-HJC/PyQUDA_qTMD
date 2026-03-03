#* Compare results of pycontract and pyquda einsum

source /lustre2/pion3d/jinchen/env/gpt.env

python compare_h5.py data/c2pt/S64T64_pyquda_frontier.c2pt.300.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S64T64_pycontract_frontier.c2pt.300.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.h5 > logs/pyp_vs_contract_2pt_src1.txt

python compare_h5.py data/c2pt/S64T64_pyquda_frontier.c2pt.300.ex.x33y37z39t49.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S64T64_pycontract_frontier.c2pt.300.ex.x33y37z39t49.1HYP_GSRC_W90_k3_T5.h5 > logs/pyp_vs_contract_2pt_src2.txt

python compare_h5.py data/qTMD/S64T64_pyquda_frontier.qTMD.300.CG.D.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 data/qTMD/S64T64_pycontract_frontier.qTMD.300.CG.D.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 > logs/pyp_vs_contract_cg_D.txt

python compare_h5.py data/qTMD/S64T64_pyquda_frontier.qTMD.300.GI_PDF.U.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 data/qTMD/S64T64_pycontract_frontier.qTMD.300.GI_PDF.U.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 > logs/pyp_vs_contract_gi_pdf_U.txt