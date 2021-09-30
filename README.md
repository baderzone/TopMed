#TopMed \
#create virtual environment:\
cd TopMed \
sh venv.sh 

source env/bin/activate

#1 full run \
#run gene_selection algorithm \
python3 gene_selection.py \
#analyze results \
python3 counts_postprocessing.py 

#if you want to run more than once, you can indicate how many by s1 flag \
python3 gene_selection.py -s1 5 \
python3 counts_postprocessing.py -s1 5

deactivate  
