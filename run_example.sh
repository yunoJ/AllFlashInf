pe_list=("16384" "32768")

for pes in ${pe_list[@]}; do
  echo $pes
  ./maestro --print_res=false \
            --print_res_csv_file=true \
            --print_log_file=false \
            --Mapping_file='data/mapping/gpt3.m' \
            --HW_file='./data/hw/pe'$pes'.m' \
            --noc_bw=32 \
            --noc_hops=1 \
            --noc_hop_latency=1 \
            --print_design_space=true \
            --msg_print_lv=0
  mv gpt3.csv gpt3_$pes.csv
  ./maestro.awk gpt3_$pes.csv
done
