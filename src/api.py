from fastapi import BackgroundTasks, FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from tempfile import NamedTemporaryFile
import aiofiles
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import asyncio
from typing import Tuple, List

from vector_index import add_to_index, search_index, load_and_index_images, load_metadata
from fast_segmentation import segment_objects, draw_boxes_and_metadata

app = FastAPI()


async def process_video_stream(temp_name: str, result_queue: asyncio.Queue, frame_queue: asyncio.Queue):
    video_stream = cv2.VideoCapture(
        filename=temp_name,
        apiPreference=1
    )
    frame_idx = 0
    results = []

    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            break
        segmented_objects = segment_objects(frame)
        frame_results = []
        objects_with_metadata = []

        for bbox, obj in segmented_objects:
            search_results = search_index(obj)
            if search_results:
                metadata = search_results[0]
                objects_with_metadata.append((obj, bbox, metadata))
                frame_results.append({"frame_index": frame_idx, "bbox": bbox, "metadata": metadata})

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
    video_stream = cv2.VideoCapture(
        filename=video_file_path,
        apiPreference=1
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


@app.post("/add_image/")
async def add_image(file: UploadFile = File(...), metadata: dict = None):
    image = Image.open(BytesIO(await file.read()))
    add_to_index(image, metadata)
    return {"message": "Image added"}


if __name__ == '__main__':
    import uvicorn
    csv_metadata = load_metadata("src/img_index/bg_master_data.csv")
    img_index_metadata = load_and_index_images(image_folder="src/img_index/smartcart_images", csv_metadata=csv_metadata)
    uvicorn.run(app, host='0.0.0.0', port=9999)
