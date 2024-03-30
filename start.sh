#!/bin/bash

echo "Step 1: Updating the VM and installing nano"
cd /home/
sudo apt update && apt-get upgrade -y
sudo apt install nano -y
echo "Update complete. Nano installed."

echo -e "\nStep 2: Exporting some aliases to bashrc"
echo -e "\n# Custom Aliases" >> ~/.bashrc
echo "alias ls='ls -lha'" >> ~/.bashrc
echo "alias cls='clear'" >> ~/.bashrc
echo "alias mamba='micromamba'" >> ~/.bashrc
echo "Aliases added to ~/.bashrc."

echo -e "\nStep 3: Creating directories"
mkdir -p /home/work/setups
echo "Directories created in /home/work/setups."

echo -e "\nStep 4: Creating an SSH key and adding it to ssh-agent"
ssh-keygen -t ed25519 -C "dhepe@perfacct.eu"
echo "SSH key generated."
echo -e "\n# Start SSH agent and add key"
eval "$(ssh-agent)"
ssh-add ~/.ssh/id_ed25519
echo "SSH key added to ssh-agent."
echo -e "\nPublic key content:"
cat ~/.ssh/id_ed25519.pub
echo -e "\nPlease copy the public key to your GitHub account."

echo -e "\nStep 5: Cloning into the work repo"
cd /home/work
git clone git@github.com:dhepeyuvi/internship.git
echo "Repository cloned into /home/work/internship."

echo -e "\nStep 6: Cloning into the reference repo"
cd /home/work
git clone git@github.com:dhepeyuvi/inference_framework_benchmark.git
echo "Repository cloned into /home/work/internship."

echo -e "\nStep 6: Installing micromamba"
cd /home/work/setups/
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
echo -e "\n# Initialize micromamba" 
/home/work/setups/bin/micromamba shell init -s bash -p /home/work/setups/micromamba
echo -e "\nSetup completed successfully! Please restart your terminal or run 'source ~/.bashrc' to apply changes."
echo -e "\nStep 7: To install a mamba env from repo run"
echo -e "\nmamba env create -f ./work/internship/environment.yml"

