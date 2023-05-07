# generate SSH with empty string as the passphrase (-N '') and write the key in .ssh/id_rsa. Pass y here-string to
# accept overwrite (y/n)? question 

ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa <<< y

key=$(<~/.ssh/id_rsa.pub)

# skip host checking for github server https://stackoverflow.com/questions/7772190/passing-ssh-options-to-git-clone
GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"

# update token regularly in the script.
curl -H "Authorization: token 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" --data '{"title":"'"$HOSTNAME"'","key":"'"$key"'"}' https://api.github.com/user/keys

sudo apt install -yq python3-pip datamash libssl-dev libz-dev luarocks python3-pip unzip

pip3 install -q crossplane pandas PyYAML numpy kubernetes scikit-optimize 

sudo luarocks install luasocket

#git clone git@github.com:gaganso/cross-layer-tuning.git

git config --global user.name "gaganso"

git config --global user.email "gagan.somashekar@gmail.com"

##ubectl apply -f metrics_server/deployment.yaml

echo "\n\n\n1. Please copy the SSH public key of the master into the worker node's authorized keys (ssh-copy-id doesn't
work as password is disabled and the private key is not on this machine)."

echo "\n2. Login to the worker node to accept the host keys for the first time (can be done when the script is run for
the first time too)"

echo "\n3. Copy volumes to BOTH volume and tmp from the laptop to test using kubectl -f apply k8s-yaml"

