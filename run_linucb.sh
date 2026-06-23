#!/bin/bash

ALPHAS=(1 3 5)

for alpha in "${ALPHAS[@]}"; do
    echo "Starting experiment with linucb_alpha=${alpha} at $(date)"

    curl -X POST http://10.220.2.139:8000/client/start \
      -H "Content-Type: application/json" \
      -d "{
        \"connect_to\": \"10.220.2.67:4433\",
        \"local_addresses\": [\"192.168.2.2\", \"10.220.2.139\"],
        \"duration\": 120,
        \"bandwidth\": \"50M\",
        \"mode\": \"uplink\",
        \"congestion_control\": \"cubic\",
        \"enable_multipath\": true,
        \"multipath_algor\": \"LinUCB\",
        \"linucb_alpha\": ${alpha},
        \"log_level\": \"INFO\"
      }"

    echo ""
    echo "Finished request for alpha=${alpha}"

    # Don't sleep after the last run
    if [ "$alpha" != "5" ]; then
        echo "Waiting 130 seconds..."
        sleep 130
    fi
done

echo "All experiments completed."
