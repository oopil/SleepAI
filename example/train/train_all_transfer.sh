# this code require gpu_id when running

mkdir runs/log
declare -a gpu_list

function get_id(){
    echo "< finding maximum ID ... >"
    rm id_list.txt
    rm id_sorted.txt

    #dir="sample"
    dir="runs/CNN_train_${organ}"
    for entry in `ls ${dir}`; do
        echo $entry >> id_list.txt
    done

    sort -V -o id_sorted.txt id_list.txt

    id=`cat id_sorted.txt | tail -n 1`
    rm id_list.txt
    rm id_sorted.txt
    echo "max id : ${id}"
    next_id="$(($id + 1))"
    echo "next id : ${next_id}"
    echo ""
}

for organ in 0 1 2 3 4;do
    get_id || next_id=1
    echo "=================================================================="
    gpu=$1
    echo "time python train.py with is_exception=True mode=train gpu_id=$gpu target=${organ} n_work=4 iter_print=False | tee runs/log/train_${organ}_ID${next_id}_exception.txt"
    time python train.py with is_exception=True mode=train gpu_id=$gpu target=${organ} n_work=4 iter_print=False | tee runs/log/train_${organ}_ID${next_id}_exception.txt

    echo "time python test.py with is_exception=True mode=test target=${organ} gpu_id=$gpu n_work=4 iter_print=False snapshot=runs/CNN_train_${organ}/${next_id}/snapshots/lowest.pth | tee runs/log/test_${organ}_ID${next_id}_exception.txt"
    time python test.py with is_exception=True mode=test target=${organ} gpu_id=$gpu n_work=4 iter_print=False snapshot=runs/CNN_train_${organ}/${next_id}/snapshots/lowest.pth | tee runs/log/test_${organ}_ID${next_id}_exception.txt

    sleep 15
done