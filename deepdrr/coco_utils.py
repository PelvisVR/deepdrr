"""Utils for the Prephix coco format."""

import numpy as np
from typing import List, Any
from pycocotools import mask as mask_utils
from itertools import groupby
import base64
import logging

from enum import Enum


log = logging.getLogger(__name__)


class SegmentationType(Enum):
    BINARY = "BINARY"
    TRAVEL = "TRAVEL"
    HITS = "HITS"


def segmentation_to_bbox(segmentation: List[List[int]]) -> List[int]:
    """Get the bounding box of a segmentation mask.

    Args:
        segmentation (List[List[int]]): The segmentation mask in the coco style.

    Returns:
        List[int]: The bounding box as [x_min, y_min, width, height]
    """
    segmentation = np.array(sum(segmentation, [])).reshape(-1, 2)
    x = segmentation[:, 0]
    y = segmentation[:, 1]
    x_min = np.min(x)
    y_min = np.min(y)
    width = np.max(x) - x_min
    height = np.max(y) - y_min
    return [int(x_min), int(y_min), int(width), int(height)]


def encode_array(array: np.ndarray) -> str:
    """Encode an array as a base64 string.

    Args:
        array (np.ndarray): The array to encode.

    Returns:
        str: The base64 encoded array.
    """
    return base64.b64encode(array.astype(np.float16).tobytes()).decode("utf-8")


def decode_array(array_str: str) -> np.ndarray:
    """Decode an array from a base64 string.

    Args:
        array_str (str): The base64 encoded array.

    Returns:
        np.ndarray: The decoded array.
    """
    return np.frombuffer(base64.b64decode(array_str), dtype=np.float16).astype(np.float32)


def make_segmentation(seg_image: np.ndarray):
    """Make a segmentation from a segmentation image.

    Args:
        seg_image (np.ndarray): The segmentation image, i.e. directly from the projector.

    Returns:
        dict: The segmentation in the adapted coco style.
    """
    binary_mask = np.array(seg_image > 0, dtype=np.uint8)
    rle_mask = mask_utils.encode(np.asfortranarray(binary_mask))
    rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
    return rle_mask


def encode_rle(mask: np.ndarray) -> str:
    """Encode a mask as a run length encoding.

    Args:
        mask (np.ndarray): The mask to encode.

    Returns:
        str: The run length encoding.
    """
    rle_dict = mask_utils.encode(np.asfortranarray(mask))
    rle = rle_dict["counts"].decode("utf-8")
    return rle


def encode_hits(hits: np.ndarray) -> dict[str, Any]:
    """Make a full segmentation from a list of hits.

    TODO: For each two hit layers (entry/exit), encode the run length encoding for the finite
    values. So each two hit layers is a separate segmentation. Will save space.

    Args:
        hits (np.ndarray): [h, w, max_hits] array of hits with a surface,
            inf denotes no more contact points.

    Returns:
        dict: The segmentation in the adapted coco style, with the following keys:
            - "size": The size of the segmentation mask.
            - "counts": List of RLE masks for each hit pair. The first mask is the segmentation.
            - "hits": The list of hit points as base64 encoded strings.
    """
    size = hits.shape[:2]
    hits = np.transpose(hits, (2, 0, 1))
    assert hits.shape[0] % 2 == 0, "Number of hit layers must be even."

    segm = dict(size=size, counts=[], hits=[])
    for i in range(0, hits.shape[0], 2):
        entry = hits[i]
        exit = hits[i + 1]
        mask = np.isfinite(entry)
        if not np.any(mask):
            if i == 0:
                segm["counts"].append(encode_rle(mask))
                segm["hits"].append("")
                segm["hits"].append("")
            break

        entry_points = entry[mask]
        exit_points = exit[mask]

        segm["counts"].append(encode_rle(mask))
        segm["hits"].append(encode_array(entry_points))
        segm["hits"].append(encode_array(exit_points))

    return segm


def detect_segmentation_type(seg: dict[str, Any]) -> SegmentationType:
    """Detect the segmentation type from a segmentation dict.

    Args:
        seg (dict[str, Any]): The segmentation dict.

    Returns:
        SegmentationType: The segmentation type.
    """
    expected_keys = {
        SegmentationType.BINARY: ["size", "counts"],
        SegmentationType.HITS: ["size", "counts", "hits"],
        SegmentationType.TRAVEL: ["size", "counts", "travel"],
    }

    for seg_type, keys in expected_keys.items():
        if set(keys) == set(seg.keys()):
            return seg_type

    raise ValueError(f"Unknown segmentation type: {seg.keys()}")


def decode_hits(seg: dict[str, Any]) -> np.ndarray:
    """Decode the hits from a segmentation.

    Args:
        seg (dict[str, Any]): the seg dict.

    Returns:
        np.ndarray: The [h,w,max_hits] array of hits with a surface,
            inf denotes no more contact points. (max_hits may vary).

    """
    hits = []
    rle: str
    for i in range(len(seg["counts"])):
        rle = seg["counts"][i]
        entry_str = seg["hits"][2 * i]
        exit_str = seg["hits"][2 * i + 1]
        mask = mask_utils.decode(dict(counts=rle, size=seg["size"])).astype(bool)
        entry_points = decode_array(entry_str)
        exit_points = decode_array(exit_str)
        entry = np.full(mask.shape, np.inf)
        exit = np.full(mask.shape, np.inf)
        entry[mask] = entry_points
        exit[mask] = exit_points
        hits.append(entry)
        hits.append(exit)

    hits = np.array(hits)
    hits = np.transpose(hits, (1, 2, 0))
    return hits


def decode_segmentation(seg: dict[str, Any]) -> np.ndarray:
    """Decode a segmentation from a dict.

    Args:
        seg (dict[str, Any]): The segmentation dict with the list of RLE masks.

    Returns:
        np.ndarray: The decoded segmentation.
    """
    if isinstance(seg["counts"], list):
        rle_dict = dict(counts=seg["counts"][0], size=seg["size"])
    else:
        rle_dict = seg
    seg = mask_utils.decode(rle_dict)
    return seg


def encode_segmentation(seg: np.ndarray) -> dict[str, Any]:
    """Encode a segmentation as a dict.

    Args:
        seg (np.ndarray): The segmentation to encode.

    Returns:
        dict[str, Any]: The encoded segmentation.
    """
    rle_dict = mask_utils.encode(np.asfortranarray(seg))
    rle_dict["counts"] = rle_dict["counts"].decode("utf-8")
    return rle_dict


def encode_travel(travel: np.ndarray) -> dict[str, Any]:
    """Encode a travel array."""
    size = travel.shape[:2]
    segm = dict(size=size, counts=[], travel=[])
    mask = travel > 0
    segm["counts"] = encode_rle(mask)
    segm["travel"] = encode_array(travel[mask])

    return segm


def decode_travel(seg: dict[str, Any]) -> np.ndarray:
    rle = seg["counts"]
    travel_points = decode_array(seg["travel"])
    mask = mask_utils.decode(dict(counts=rle, size=seg["size"])).astype(bool)
    travel = np.zeros(mask.shape, dtype=np.float32)
    travel[mask] = travel_points

    return travel


def decode_mask_travel(seg: dict[str, Any]) -> np.ndarray:
    rle = seg["counts"]
    travel_points = decode_array(seg["travel"])
    mask = mask_utils.decode(dict(counts=rle, size=seg["size"])).astype(bool)
    travel = np.zeros(mask.shape, dtype=np.float32)
    travel[mask] = travel_points

    return mask, travel


def area(seg: dict[str, Any]) -> float:
    """Get the area of a segmentation.

    Args:
        seg (dict[str, Any]): The segmentation dict.

    Returns:
        float: The area of the segmentation.
    """
    counts = seg["counts"]
    if isinstance(counts, list):
        counts = counts[0]
    return mask_utils.area(dict(size=seg["size"], counts=counts))


def toBbox(seg: dict[str, Any]) -> List[int]:
    """Get the bounding box of a segmentation.

    Args:
        seg (dict[str, Any]): The segmentation dict.

    Returns:
        List[int]: The bounding box as [x_min, y_min, width, height]
    """
    counts = seg["counts"]
    if isinstance(counts, list):
        counts = counts[0]
    return mask_utils.toBbox(dict(size=seg["size"], counts=counts))


def compute_travel(hits: np.ndarray) -> np.ndarray:
    """Compute the travel distance of the hits.

    Args:
        hits (np.ndarray): The [h,w,max_hits] array of hits with a surface,
            inf denotes no more contact points. (max_hits may vary).

    Returns:
        np.ndarray: The [h,w] array of travel distances.
    """
    travel = np.zeros(hits.shape[:2], dtype=np.float32)
    for i in range(0, hits.shape[2], 2):
        entries = hits[:, :, i]
        exits = hits[:, :, i + 1]
        mask = np.isfinite(entries)
        if not np.any(mask):
            break

        travel[mask] += exits[mask] - entries[mask]

    return travel


def box_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert the boxes from xywh to xyxy format.

    Args:
        boxes (np.ndarray): The boxes to convert, in the xywh format.

    Returns:
        np.ndarray: The boxes in the xyxy format.
    """
    return np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]], axis=1)


def box_fliph_xyxy(boxes: np.ndarray, w: float = 1) -> np.ndarray:
    """Flip the boxes horizontally.

    Args:
        boxes (np.ndarray): The boxes to flip, in the (x_min, y_min, x_max, y_max) format.
        w (int): The width of the image.

    Returns:
        np.ndarray: The flipped boxes.
    """

    return np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], axis=1)


def mask_to_rle(mask: np.ndarray):
    binary_mask = (mask > 0).astype(np.uint8)
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts: list[int] = rle.get("counts")

    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle
