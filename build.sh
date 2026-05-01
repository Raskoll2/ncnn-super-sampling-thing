cd ~/repos/SuperSampling/build
rm -rf *
cmake ..
make -j$(nproc)
cp ./upscaler ../upscaler
