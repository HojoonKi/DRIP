docker run -dit\
	--gpus all\
	-p 5656:8899\
	-v $(pwd):/workspace\
	-v /mnt/nfs_shared_data/dataset:/workspace/dataset \
	--ipc=host\
	--user root\
	--name drip\
	drip:latest\
	bash