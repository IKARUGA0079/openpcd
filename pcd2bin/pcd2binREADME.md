# ifrtp
# ifr_Teamprojekt_SS2022
cd ~/Downloads/ifR
git clone https://github.com/ZWJcoconut/ifrtp.git

#bag2pcd
roscore
rosrun pcl_ros bag_to_pcd ros.bag /velodyne_points ./pcd

#pcd2bin
pip install numpy
pip install argparse
pip install pypcd
pip install tqdm

cd ifrtp
python pcd2bin.py --pcd_path=../pcd --bin_path=../bin --file_name=velodyne

