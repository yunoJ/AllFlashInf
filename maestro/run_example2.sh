./maestro --print_res=true \
          --print_res_csv_file=true \
          --print_log_file=false \
          --Mapping_file='data/mapping/Transformer_Complete.m' \
          --HW_file='./data/hw/pe16384.m' \
          --noc_bw=32 \
          --noc_hops=1 \
          --noc_hop_latency=1 \
          --l1_size=256000 \
          --l2_size=20480000 \
          --num_pes=256 \
          --print_design_space=true \
          --msg_print_lv=0

