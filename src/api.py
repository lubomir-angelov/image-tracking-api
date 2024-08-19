import base64
from fastapi import BackgroundTasks, FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from tempfile import NamedTemporaryFile
import aiofiles
import glob
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import asyncio
from typing import Tuple, List
import uvicorn

from similarity_search import Similarity
from vector_index import load_and_index_images, load_metadata
from fast_segmentation import FastSegment

model = FastSegment()
csv_metadata = load_metadata("src/img_index/bg_master_data.csv")
img_index_metadata = load_and_index_images(image_folder="src/img_index/smartcart_images", csv_metadata=csv_metadata)
sim = Similarity(external_metadata=img_index_metadata)


def create_application() -> FastAPI:
    application = FastAPI()

    return application


app = create_application()

def draw_boxes_and_metadata(frame, objects_with_metadata):
    for obj, bbox, metadata in objects_with_metadata:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Metadata: {metadata}"
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def segment_image(image):
    masks = model.get_masks(image=image)
    boxes = model.get_boxes()

    return masks, boxes

async def process_video_stream(temp_name: str, result_queue: asyncio.Queue, frame_queue: asyncio.Queue):
    # apiPreference 0 is FFMPEG!
    video_stream = cv2.VideoCapture(
        filename=temp_name,
        apiPreference=0
    )
    frame_idx = 0
    results = []

    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            break
        frame_results = []
        objects_with_metadata = []

        masks, boxes = segment_image(image=frame)

        if boxes is not None:
            for box, cls in zip(boxes, masks):
                bbox = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
                #image.convert("RGB")
                img = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                scores, search_results = sim.get_neighbors(img)
                #search_results = search_index(img)

                if search_results:
                    metadata = search_results["additional_info"]
                    objects_with_metadata.append((img, bbox, metadata))
                    frame_results.append({"frame_index": frame_idx, "bbox": bbox, "metadata": metadata, "score": scores})

        results.append(frame_results)
        frame_with_metadata = draw_boxes_and_metadata(frame, objects_with_metadata)
        await frame_queue.put((frame_with_metadata, frame_idx))
        frame_idx += 1

    video_stream.release()
    await frame_queue.put(None)  # Signal that the stream is finished
    await result_queue.put(results)

async def save_video(output_path, frame_queue, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        frame_data = await frame_queue.get()
        if frame_data is None:
            break
        frame, _ = frame_data
        out.write(frame)

    out.release()


def get_fps_and_frame_size(video_file_path: str) -> Tuple[float, int]:
    # Read initial frame to get FPS and frame size
    # apiPreference 0 is FFMPEG!
    video_stream = cv2.VideoCapture(
        filename=video_file_path,
        apiPreference=0
    )
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_stream.release()

    return fps, frame_size


@app.post("/upload_video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    frame_queue = asyncio.Queue()
    result_queue = asyncio.Queue()

    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            try:
                contents = await file.read()
                await temp.write(contents)
            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                await file.close()
        fps, frame_size = get_fps_and_frame_size(video_file_path=temp.name)

        # Start processing video stream in a separate thread
        processing_task = asyncio.create_task(
            process_video_stream(
                temp_name=temp.name,
                result_queue=result_queue, 
                frame_queue=frame_queue
            ))

        # Start video saving in a separate thread
        output_video_path = "output/processed_video.mp4"
        background_tasks.add_task(save_video, output_video_path, frame_queue, fps, frame_size)

        # Wait for processing to finish and get results
        results = await result_queue.get()
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)
    
    return {"results": results}


@app.post("/segment_image/")
async def add_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    masks, boxes = segment_image(image=image)

    # round up to 2 precison points
    boxes = np.around(boxes,2).tolist()

    boxes_serialized = {"boxes": []};
    #encoded_image = base64.b64encode(await file.read())
    # file = await file.read()
    #nparr = np.frombuffer(file, np.uint8)
    # Convert the NumPy array into an OpenCV image
    #decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    decoded_image = np.array(image)

    for box in boxes:
        img = decoded_image[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
        scores, search_results = sim.get_neighbors(img)
        score = scores[0:1][0]
        #search_results = search_index(img)

        if search_results:
            if score < 25:
            #metadata = search_results["additional_info"]
                item_name = os.path.basename(search_results["image_path"][0])

                boxes_serialized["boxes"].append(
                    {
                        "item_name": item_name.split('.')[0], # remove .jpg from output
                        "score": float(score),
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                    }
                )
        else:
            boxes_serialized["boxes"].append(
                {
                    "item_name": "no item found",
                    "score": 0,
                    "x1": 0,
                    "y1": 0,
                    "x2": 0,
                    "y2": 0,
                }
            )

    return boxes_serialized

@app.get("/play_video")
async def video_endpoint():
    def iterfile():
        with open("output/video.mp4", mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="video/mp4")

def cleanup_folder():
    files = glob.glob('output_images/*')
    for f in files:
        os.remove(f)

def write_video_from_images():
    images = []
    files = glob.glob('output_images/*')
    if files:
        for file in files:
            img = Image.open(file)
            images.append(cv2.imread(file))

        video = cv2.VideoWriter('output/video.avi',-1,1,(img.width,img.height))

        for image in images:
            video.write(image)
        
    return


@app.post("/add_image/")
async def add_image(file: UploadFile = File(...), metadata: dict = None):
    image = Image.open(BytesIO(await file.read()))
    #add_to_index(image, metadata)
    return {"message": "Image added"}


if __name__ == "__main__":
    
    #write_video_from_images()
    #cleanup_folder() 
    uvicorn.run("src.api:app", host='0.0.0.0', port=9999, reload=True)

