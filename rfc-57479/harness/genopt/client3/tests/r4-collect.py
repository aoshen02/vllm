import glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + "/*/result.json")):
    r = json.load(open(f)); gb = sum(r["response_bytes"]) / 1e9
    print(f.split("/")[-2], r["status"], "GB %.1f recv %.2f parsed %.2f GBps %.2f cpu %.1f rss %.1f scratch_peak_MB %.1f" % (
        gb, r["all_bodies_received_s"], r["all_parsed_s"], gb / r["all_parsed_s"], r["client_process_cpu_s"], r["client_max_rss_gib"],
        r["v3_parse_scratch_peak_bytes_max"] / 1e6))
