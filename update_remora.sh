git fetch remora
git checkout remora/master
git subtree split -P include/remora/ -b temporary
git checkout ${1:-master}
git subtree merge --squash -P include/shark/LinAlg/BLAS/ temporary
git branch -D temporary