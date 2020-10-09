#! /usr/bin/awk -f

BEGIN {
  FS=", "
  # columns = "Runtime (Cycles),input l2 write,filter l2 write,Compute Delay (Avg)"
  columns = "Runtime (Cycles),input l2 write,filter l2 write,Avg number of utilized PEs"
  n = split(columns,out,",")
  abw = 0;
  max_bw = 0;
}
# NR>1 {
#   for (i in out) {
#     # printf "%s: %s\n", out[i], $ix[out[i]]
#     run += $ix[out[i]]
#   }
#   print run
#   # print ""
# }
NR==1 {
  for (i=1; i<=NF; i++) {
    ix[$i] = i
  }
}
{
  run += $ix[out[1]];
  data += ($ix[out[2]]+$ix[out[3]])*2;
  pes += $ix[out[4]];
  # Bytes / nanosecons = GB/s
  bw = ($ix[out[2]]+$ix[out[3]])*2/($ix[out[1]]*1.25);
  if (max_bw < bw)
    max_bw = bw;
  # printf "Required BW: %s", bw;
  # for (i in out) {
  #   printf "%s, ", $ix[out[i]];
  # }
  # print OFD;
}

END {
  print "Runtime:", run;
  # print "Average BW:", data/(run*1.25);
  # print "Max BW:", max_bw;
  # print "Avg util PEs:", pes/NR;
}
