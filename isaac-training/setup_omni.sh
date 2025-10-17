cd third_party/OmniDrones
cp -r conda_setup/etc $CONDA_PREFIX

echo -e "\033[0;31mSetting up OmniDrones package...\033[0m"
cd ../OmniDrones
pip install -e .