from tools.infer import *
def runNew(FLAGS, cfg: dict):
    if FLAGS['rtn_im_file']:
        cfg['TestReader']['sample_transforms'][0]['Decode'][
            'rtn_im_file'] = FLAGS['rtn_im_file']
    ssod_method = cfg.get('ssod_method', None)
    if ssod_method == 'ARSL':
        trainer = Trainer_ARSL(cfg, mode='test')
        trainer.load_weights(cfg.weights, ARSL_eval=True)
    else:
        trainer = Trainer(cfg, mode='test')
        trainer.load_weights(cfg.weights)
    # get inference images
    if FLAGS['do_eval']:
        dataset = create('TestDataset')()
        images = dataset.get_images()
    else:
        images = get_test_images(FLAGS['infer_dir'], FLAGS['infer_img'], FLAGS['infer_list'])

    # inference
    if FLAGS['slice_infer']:
        trainer.slice_predict(
            images,
            slice_size=FLAGS['slice_size'],
            overlap_ratio=FLAGS['overlap_ratio'],
            combine_method=FLAGS['combine_method'],
            match_threshold=FLAGS['match_threshold'],
            match_metric=FLAGS['match_metric'],
            draw_threshold=FLAGS['draw_threshold'],
            output_dir=FLAGS['output_dir'],
            save_results=FLAGS['save_results'],
            visualize=FLAGS['visualize'])
    else:
        results = trainer.predict(
            images,
            draw_threshold=FLAGS['draw_threshold'],
            output_dir=FLAGS['output_dir'],
            save_results=FLAGS['save_results'],
            visualize=FLAGS['visualize'],
            save_threshold=FLAGS['save_threshold'],
            do_eval=FLAGS['do_eval'])
        
        return results

def merge_argsNew(config, args, exclude_args=['config', 'opt', 'slim_config']):
    for k, v in args.items():
        if k not in exclude_args:
            config[k] = v
    return config

if __name__ == '__main__':
    
    FLAGS = {"config":'./configs/ssd/ssdlite_mobilenet_v3_small_320_coco.yml',
    "opt":{'weights': './output/ssd/best_model.pdparams'},
    "infer_dir":'./tt_final',
    "infer_list":None,
    "infer_img":None,
    "output_dir":'infer_output/ssd/',
    "draw_threshold":0.5,
    "save_threshold":0.5,
    "slim_config":None,
    "use_vdl":False,
    "do_eval":False,
    "vdl_log_dir":'vdl_log_dir/image',
    "save_results":False,
    "slice_infer":False,
    "slice_size":[640, 640],
    "overlap_ratio":[0.25, 0.25],
    "combine_method":'nms',
    "match_threshold":0.6,
    "match_metric":'ios',
    "visualize":True,
    "rtn_im_file":False
    }
    cfg = load_config(FLAGS['config'])
    merge_argsNew(cfg, FLAGS)
    merge_config(FLAGS['opt'])

    result = runNew(FLAGS, cfg)
