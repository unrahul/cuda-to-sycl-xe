#!/usr/bin/env bash
# dev_campaign.sh PACKAGE TAG LANE...   LANE = renderD12N:TASK[,TASK...]
# Paired development campaign: for each task, the skill and control arms run at the same time on
# that lane's GPU (assessments serialize on the GPU's lock); a lane's tasks run one after another.
set -u
cd "$(dirname "$0")"
[[ $# -ge 3 ]] || { echo 'usage: dev_campaign.sh PACKAGE TAG LANE...' >&2; exit 2; }
pkg=$1 tag=$2; shift 2
mkdir -p runs/dev-logs
for lane in "$@"; do
    gpu=${lane%%:*} tasks=${lane#*:}
    (
        for task in ${tasks//,/ }; do
            for arm in skill control; do
                extra=()
                [[ $arm == skill ]] && extra=(--package "$pkg")
                python3 dev_episode.py "$task" --arm "$arm" --runner codex --gpu "$gpu" "${extra[@]}" \
                    > "runs/dev-logs/$tag-${task,,}-$arm.log" 2>&1 &
                sleep 2
            done
            wait
        done
    ) &
done
wait
