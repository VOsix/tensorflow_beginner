#!/bin/bash

source ~/.bashrc

export HADOOP_USER_NAME="lzjiang"

ps_hosts=$1
worker_hosts=$2
task_type=$3
task_index=$4
ps_num=$5
worker_num=$6

feature_size=100000
eval_ckpt_id=0

hour=$(echo $(date +"%H") | awk -F ' ' '{print $0+0}')
current_time=$(date +"%Y-%m-%d %H:%M:%S")

index=""

# get hdfs name
hdfs_name_list=("hdfs://sparktrain:8020" "hdfs://10.20.2.6:8020" "hdfs://10.20.2.5:8020")
hdfs_name="${hdfs_name_list[0]}"
hdfs_prefix="${hdfs_name}/data/project/dataming/ads_prd/LZ_dnn_datav1"
	
done_path="${hdfs_prefix}/done_*"
train_data=""
train_data_list=()
eval_data=""

model_version="_news_indicator"
model_dir="${hdfs_prefix}/model${model_version}"
export_savedmodel="${hdfs_prefix}/export_savedmodel${model_version}"
savedmodel_mode="parsing_new"

function print() {
    local msg="$1"
    local prefix=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "\e[032m[${prefix}][INFO] ${msg}\e[0m"
    return 0
}


function check_return() {
    local ret=$1
    local prefix=$(date +"%Y-%m-%d %H:%M:%S")
    if [[ ${ret} -eq 0 ]]; then
        echo -e "\e[032m[${prefix}][INFO] success\e[0m"
    else
        echo -e "\e[031m[${prefix}][ERROR] failed\e[0m"
        exit ${ret}
    fi
    return ${ret}
}


function get_train_data() {
    local max_retries=5
    for ((retry=0; retry<${max_retries}; ++retry)); do
        local train_path_list=($(hadoop fs -cat ${done_path}/*))
        [[ $? -eq 0 ]] && break
        sleep 6
    done

    train_data=""
    for ((i=0; i<${#train_path_list[@]}; ++i)); do
        local train_path=${train_path_list[$i]}
        if [[ ${#train_path} -gt 0 ]]; then
            if [[ ${#train_data} -eq 0 ]]; then
                train_data="${hdfs_name}${train_path}/*"
            else
                train_data="${train_data} ${hdfs_name}${train_path}/*"
            fi
        fi
    done

    if [[ ${#train_data} -gt 0 ]]; then
        print "train data: ${train_data}"
    else
        print "failed to get train data"
        check_return 1
    fi
}


function distribute_files() {
    #local task_type=$1
    #if [[ "${task_type}" != "worker" ]]; then
    #    train_data_list=("${train_data}/*")
    #    return 0
    #fi

    local files=($(hadoop fs -ls -R ${train_data} | awk -F ' ' '{if(NF == 8 && $5 > 0) print $8}'))
    print "there are ${#files[@]} files in ${train_data}"
    if [[ ${#files[@]} -lt ${worker_num} ]]; then
        print "worker num must be not greater than files num"
        check_return 1
    fi

    set +x
    for ((i=0; i<${#files[@]}; ++i)); do
        let j=i%${worker_num}
        if [[ ${i} -lt ${worker_num} ]]; then
            train_data_list[j]="${files[$i]}"
        else
            train_data_list[j]="${train_data_list[$j]},${files[$i]}"
        fi
    done
    set -x
    check_return $?
}


function run_task() {
    local task_type=$1
    local task_index=$2

    python wnd_concat.py \
            --run_mode "distributed" \
            --job_type "train" \
            --model_type "wide_and_deep" \
            --model_dir "${model_dir}" \
            --feature_size ${feature_size} \
            --cold_start False \
            --train_data "${train_data_list[${task_index}]}" \
            --worker_hosts "${worker_hosts}" \
            --ps_hosts "${ps_hosts}" \
            --task_type "${task_type}" \
            --task_index ${task_index}
    local ret=$?
    return ${ret}
}


function export_savedmodel() {
    python wnd_concat.py \
            --run_mode "local" \
            --job_type "export_savedmodel" \
            --model_type "wide_and_deep" \
            --model_dir "${model_dir}" \
            --feature_size ${feature_size} \
            --export_savedmodel "${export_savedmodel}" \
            --savedmodel_mode "${savedmodel_mode}"
    local ret=$?
    return ${ret}
}


function eval_model() {
    python wnd_concat.py \
            --run_mode "local" \
            --job_type "eval" \
            --model_type "wide_and_deep" \
            --model_dir "${model_dir}" \
            --feature_size ${feature_size} \
            --eval_ckpt_id ${eval_ckpt_id} \
            --eval_data "${eval_data}"
    local ret=$?
    return ${ret}
}

print "ps_hosts: ${ps_hosts}, worker_hosts: ${worker_hosts}, task_type: ${task_type}, task_index: ${task_index}, ps_num: ${ps_num}, worker_num: ${worker_num}"
print "current_time: ${current_time}"
print "hour: ${hour}"
hadoop fs -chmod -R 755 ${model_dir}

if [ $hour -gt 6 ]||[ $hour -eq 0 ]; then
    print "$hour is bigger than 6 or $hour is equal 0 , sleep 10s"
    sleep 10
    exit 0
fi

if [[ "${task_type}" == "worker" ]]; then
    print "${worker_num} worker tasks will start..."
    get_train_data
    distribute_files ${task_type}
    run_task ${task_type} ${task_index}
elif [[ "${task_type}" == "ps" ]]; then
    print "${ps_num} ps tasks will start..."
    get_train_data
    distribute_files ${task_type}
    run_task ${task_type} ${task_index}
else
    print "unsupported task type: ${task_type}"
    check_return 1
fi
print "Train Mode: task ${task_type} completed"

