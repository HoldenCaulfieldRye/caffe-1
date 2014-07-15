# NOT READY YET

cd /data/ad6813
git clone --depth=1 --branch master git://github.com/BVLC/caffe.git
source /usr/local/cuda/setup.sh


if [ ! -d /data/`whoami`/CUDA-SDK ]
   then
       echo "copying CUDA-SDK to /data/`whoami`..."
       mkdir -p /data/`whoami`/CUDA-SDK
       cd !$
       cp -pr /usr/local/cuda/gpu_sdk/{C,shared} .
       cd shared
       make
       cd ../C/common
       make
fi

cd /data/`whoami`

# 1) check version of boost with pipe-classification/caffe/install/??

# 2) check if boost dir exists
if [ ! -d boost ]
   then

mkdir boost
cd boost
wget -O boost_1_55_0.tar.gz
tar zxvf boost_1_55_0.tar.gz
cd boost_1_55_0

if [ ! -d /homes/`whoami`/.local ]
   then
       echo "creating ~/.local"
       mkdir -p /homes/`whoami`/.local
       
fi



./b2 install --prefix=$HOME/.local
# add path environment variables
echo "export BOOST_ROOT=$HOME/.local"
       
