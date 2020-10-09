pe_list=("16384" "32768")

for pe in ${pe_list[@]}; do
  ./maestro --print_res=true \
            --print_res_csv_file=true \
            --print_log_file=false \
            --Mapping_file='data/mapping/gpt3.m' \
            --HW_file='./data/hw/pe'$pe'.m' \
            --noc_bw=32 \
            --noc_hops=1 \
            --noc_hop_latency=1 \
            --l1_size=256000 \
            --l2_size=20480000 \
            --num_pes=256 \
            --print_design_space=true \
            --msg_print_lv=0
  mv gpt3.csv gpt3_$pe.csv
  ./maestro.awk gpt3_$pe.csv
done
