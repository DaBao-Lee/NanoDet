from demo.demo import *

class PredictResult:
    def __init__(self):
        self.result = {}
    
    def add_result(self, image_name, meta, res):

        self.result[image_name] = {
            "meta": meta,
            "res": res
        }
def run(
    demo="image",
    config="./config/demo.yml",
    model="./model/demo.pth",
    path="./demo",
    camid=0,
    save_result=True,
    show_result=True,
):
    result = PredictResult()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, model, logger, device="cuda:0")
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()

    if demo == "image":
        files = get_image_list(path) if os.path.isdir(path) else [path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            result.add_result(image_name, meta, res)
            if save_result:
                result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
            if show_result:
                ch = cv2.waitKey(0)
                if ch in [27, ord("q"), ord("Q")]:
                    break

        return result
    
    elif demo in ["video", "webcam"]:
        cap = cv2.VideoCapture(path if demo == "video" else camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank, save_folder)
        save_path = (
            os.path.join(save_folder, path.replace("\\", "/").split("/")[-1])
            if demo == "video"
            else os.path.join(save_folder, "camera.mp4")
        )
        print(f"save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                break
            meta, res = predictor.inference(frame)
            result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
            if save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch in [27, ord("q"), ord("Q")]:
                break