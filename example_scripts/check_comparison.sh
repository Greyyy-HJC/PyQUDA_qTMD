#* Compare results on Aurora and LQ2

python compare_h5.py LQ2/data/c2pt/S64T64_gpt_lq2.c2pt.1050.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.h5 Aurora/data/c2pt/S64T64_pyquda_aurora.c2pt.1050.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.h5 > logs/gpt_lq2_vs_pyquda_aurora_2pt.txt

python compare_h5.py LQ2/data/qTMD/S64T64_gpt_lq2.qTMD.1050.CG.D.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 Aurora/data/qTMD/S64T64_pyquda_aurora.qTMD.1050.CG.D.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 > logs/gpt_lq2_vs_pyquda_aurora_3pt_CG.txt

python compare_h5.py LQ2/data/qTMD/S64T64_gpt_lq2.qTMD.1050.GI_PDF.D.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 Aurora/data/qTMD/S64T64_pyquda_aurora.qTMD.1050.GI_PDF.D.ex.x33y37z39t1.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 > logs/gpt_lq2_vs_pyquda_aurora_3pt_GI.txt