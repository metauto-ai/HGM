# remove all docker containers, including the running ones
# docker rm -f $(docker ps -a -q)

# remove all unused docker images and containers
docker system prune -f

# Initialize conda for the current shell and activate environment
eval "$(conda shell.bash hook)"
conda activate agent

python hgm.py