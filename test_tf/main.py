import logging.config

from module.imageprocess import images_list, load_images
from module.predict import predict_model, predict_result
from module.roi import setting_roi


def run(source_path, model_path, categories, show=None):
    logging.config.fileConfig("logger.conf")
    logger = logging.getLogger("Tester")

    classes = len(categories)
    count = 0

    logger.info(f"Image List Load")
    img_list = images_list(source_path=source_path)
    logger.info(f"Image List Load Done.")

    roi_name = "53R_201"
    logger.info(f"ROI Setting: {roi_name}")
    roi = setting_roi(roi_name)
    logger.info(f"Done.")

    logger.info(f"Image predict start")

    predict_list = []
    for img_path in img_list:
        img = load_images(path=img_path, roi=roi)
        filename = img_path.split("/")[-1].split(".")[0]
        pred_value = predict_model(model_path=model_path, load_img=img)

        for i in range(classes):
            predict_list.append(pred_value.count(i))

        count = predict_result(pred_list=predict_list, filename=filename, categories=categories, show=show)

    logger.info(f"정확도: {count / len(img_list) * 100:.2f}%")

    logger.info(f"Image predict done.")


if __name__ == "__main__":
    SOURCE_PATH = "/Users/methodfunc/Pictures/wet&moist"
    MODEL_PATH = "/Users/methodfunc/models/"

    CATEGORIES = ["건조_주간", "건조_야간", "습윤_주간", "습윤_야간"]
    SHOW = 0

    run(model_path=MODEL_PATH, source_path=SOURCE_PATH, categories=CATEGORIES, show=SHOW)
