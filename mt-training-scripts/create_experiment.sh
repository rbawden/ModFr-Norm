#!/bin/sh

thisdir=`dirname $0`

optspec=":-:"
while getopts "$optspec" optchar; do
    case "${optchar}" in
        -)
            case "${OPTARG}" in
		datatype)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    datatype=$val
                    ;;
                datapath)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    datapath=$val
                    ;;
		modeldir)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    modeldir=$val
                    ;;
		seed)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    seed=$val
                    ;;
		arch)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    arch=$val
                    ;;
		encoderlayers)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    encoderlayers=$val
                    ;;
		decoderlayers)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    decoderlayers=$val
                    ;;
		attnheads)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    attnheads=$val
                    ;;
		embeddim)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    embeddim=$val
                    ;;
		ffdim)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    ffdim=$val
                    ;;
		hidden)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    hidden=$val
                    ;;
		dropout)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    dropout=$val
                    ;;
		lr)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    lr=$val
                    ;;
		bsz)
                    val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    echo "Parsing option: '--${OPTARG}', value: '${val}'" >&2;
		    bsz=$val
                    ;;
                *)
                    if [ "$OPTERR" = 1 ] && [ "${optspec:0:1}" != ":" ]; then
                        echo "Unknown option --${OPTARG}" >&2
                    fi
                    ;;
            esac;;
    esac
done

# create the directory
[ -d $modeldir ] || mkdir $modeldir
[ -d $modeldir/$seed ] || mkdir $modeldir/$seed 

# copy the contents
cp $thisdir/experiment_templates/*.sh $modeldir/$seed/

# replace the variables
perl -i -pe "s/<arch>/$arch/g" $modeldir/$seed/*.sh
perl -i -pe "s/<encoderlayers>/$encoderlayers/g" $modeldir/$seed/*.sh
perl -i -pe "s/<decoderlayers>/$decoderlayers/g" $modeldir/$seed/*.sh
perl -i -pe "s/<attnheads>/$attnheads/g" $modeldir/$seed/*.sh
perl -i -pe "s/<embeddim>/$embeddim/g" $modeldir/$seed/*.sh
perl -i -pe "s/<ffdim>/$ffdim/g" $modeldir/$seed/*.sh
perl -i -pe "s/<dropout>/$dropout/g" $modeldir/$seed/*.sh
perl -i -pe "s/<lr>/$lr/g" $modeldir/$seed/*.sh
perl -i -pe "s/<hidden>/$hidden/g" $modeldir/$seed/*.sh
perl -i -pe "s/<seed>/$seed/g" $modeldir/$seed/*.sh
perl -i -pe "s/<bsz>/$bsz/g" $modeldir/$seed/*.sh

datapath_esc=`echo $datapath | perl -pe 's/\//\\\\\//g'`

perl -i -pe "s/<datapath>/$datapath_esc/g" $modeldir/$seed/*.sh

# copy the model details
[ ! -f $modeldir/$seed/model_details.list ] || rm $modeldir/$seed/model_details.list
echo "datatype=$datatype" >> $modeldir/$seed/model_details.list
echo "arch=$arch" >> $modeldir/$seed/model_details.list
echo "encoderlayers=$encoderlayers" >> $modeldir/$seed/model_details.list
echo "decoderlayers=$decoderlayers" >> $modeldir/$seed/model_details.list
echo "attnheads=$attnheads" >> $modeldir/$seed/model_details.list
echo "embeddim=$embeddim" >> $modeldir/$seed/model_details.list
echo "ffdim=$ffdim" >> $modeldir/$seed/model_details.list
echo "dropout=$dropout" >> $modeldir/$seed/model_details.list
echo "lr=$lr" >> $modeldir/$seed/model_details.list
echo "bsz=$bsz" >> $modeldir/$seed/model_details.list
echo "seed=$seed" >> $modeldir/$seed/model_details.list
echo "hidden=$hidden" >> $modeldir/$seed/model_details.list



