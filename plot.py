from tensorflow.python.summary.summary_iterator import summary_iterator

if __name__ == '__main__':
    a = []
    for summary in summary_iterator("./i3d_dir/Logits/sub01/EBA/events.out.tfevents.1638286672.HELPeR"):
        if summary.summary.value:
            if summary.summary.value[0].tag == "Correlation/val":
                print(summary.summary.value[0].simple_value)
                a.append(summary.summary.value[0].simple_value)
    print(len(a))
