"""Sequential scene reads and resumable frozen hierarchy caches with native-point mapping."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + '.part')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(partial, path)


def array_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode())
    digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return digest.hexdigest()


def voxel_keys(coords, grid_size, backend='cpu'):
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError('Invalid coordinates')
    origin = coords.min(axis=0)
    keys = np.empty(len(coords), dtype=np.uint64)
    for start in range(0, len(coords), 1_000_000):
        stop = min(start + 1_000_000, len(coords))
        grid = np.floor((coords[start:stop] - origin) / grid_size).astype(np.uint64)
        if grid.max() >= 2**21:
            raise ValueError('Coordinate grid exceeds reversible 21-bit key; no silent hash collisions')
        keys[start:stop] = (grid[:, 0] << np.uint64(42)) | (grid[:, 1] << np.uint64(21)) | grid[:, 2]
    if backend == 'cuda':
        import torch
        gpu_keys = torch.from_numpy(keys.view(np.int64)).cuda()
        gpu_unique, gpu_inverse = torch.unique(gpu_keys, sorted=True, return_inverse=True)
        gpu_representatives = torch.full((len(gpu_unique),), len(keys), dtype=torch.long, device='cuda')
        gpu_representatives.scatter_reduce_(0, gpu_inverse, torch.arange(len(keys), device='cuda'),
                                           reduce='amin', include_self=True)
        unique = gpu_unique.cpu().numpy().astype(np.uint64)
        representatives = gpu_representatives.cpu().numpy()
        inverse = gpu_inverse.int().cpu().numpy()
    elif backend == 'cpu':
        unique, representatives, inverse = np.unique(keys, return_index=True, return_inverse=True)
    else:
        raise ValueError('Unknown voxel uniqueness backend')
    del keys
    return unique, representatives, inverse.astype(np.int32)


def ensure_scene(raw_root, scene, cache_root, encoder, cache_id, block_size, heartbeat, grid_size=0.02, storage_guard=lambda extra=0: None):
    import torch
    from head import frozen_hierarchy
    raw = Path(raw_root) / scene
    folder = Path(cache_root) / scene
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / 'SCENE_COMPLETE.json'
    identity = {name: {'bytes': (raw / name).stat().st_size, 'mtime_ns': (raw / name).stat().st_mtime_ns}
                for name in ('coord.npy', 'color.npy', 'normal.npy', 'class.npy')}
    if complete.exists():
        existing = json.loads(complete.read_text())
        if existing['cache_id'] != cache_id or existing['source_identity'] != identity:
            raise ValueError('Incompatible scene cache; use a new cache namespace')
        if not all((folder / name).is_file() for name in existing['blocks'] + ['inverse.npy', 'labels.npy']):
            raise ValueError('Completed cache is missing artifacts')
        return existing
    heartbeat(stage='scene_read', scene=scene)
    start = time.perf_counter()
    coords = np.load(raw / 'coord.npy', allow_pickle=False)
    rgb = np.load(raw / 'color.npy', allow_pickle=False)
    normals = np.load(raw / 'normal.npy', allow_pickle=False)
    labels = np.load(raw / 'class.npy', allow_pickle=False)
    if len({len(x) for x in (coords, rgb, normals, labels)}) != 1 or labels.ndim != 1:
        raise ValueError('Input alignment failure')
    if np.any(labels > 17) or np.any(labels < 0) or not np.isfinite(normals).all():
        raise ValueError('Invalid labels or normals')
    hashes = {name: array_hash(value) for name, value in
              [('coord', coords), ('color', rgb), ('normal', normals), ('class', labels)]}
    load_seconds = time.perf_counter() - start
    heartbeat(stage='scene_voxelize', scene=scene, raw_points=len(coords), read_seconds=load_seconds)
    start = time.perf_counter()
    keys, representatives, inverse = voxel_keys(coords, grid_size, backend='cuda')
    storage_guard(5*len(coords) + 1024*len(keys))
    np.save(folder / 'labels.npy.part', labels.astype(np.uint8), allow_pickle=False)
    os.replace(folder / 'labels.npy.part.npy', folder / 'labels.npy')
    np.save(folder / 'inverse.npy.part', inverse, allow_pickle=False)
    os.replace(folder / 'inverse.npy.part.npy', folder / 'inverse.npy')
    voxel_seconds = time.perf_counter() - start
    raw_points, voxel_count = len(coords), len(keys)
    blocks, channels = [], None
    encoder_seconds = 0.
    for block_number, start in enumerate(range(0, voxel_count, block_size)):
        stop = min(start + block_size, voxel_count)
        name = f'block_{block_number:05d}.pth'
        target = folder / name
        blocks.append(name)
        if target.exists():
            saved = torch.load(target, weights_only=True, map_location='cpu')
            if saved.get('cache_id') != cache_id or saved['start'] != start or saved['stop'] != stop:
                raise ValueError('Partial cache mismatch')
            channels = saved['channels']
            continue
        selected = representatives[start:stop]
        xyz = coords[selected].astype(np.float32)
        center = (xyz.min(0) + xyz.max(0)) / 2
        center[2] = xyz[:, 2].min()
        xyz -= center
        encoded_keys = keys[start:stop]
        grid = np.stack([encoded_keys >> np.uint64(42),
                         (encoded_keys >> np.uint64(21)) & np.uint64(2**21-1),
                         encoded_keys & np.uint64(2**21-1)], axis=1).astype(np.int32)
        grid -= grid.min(0)
        feat = np.concatenate([xyz, rgb[selected].astype(np.float32)/127.5-1., normals[selected]], axis=1).astype(np.float32)
        point = {'coord': torch.from_numpy(xyz).cuda(), 'grid_coord': torch.from_numpy(grid).cuda(),
                 'feat': torch.from_numpy(feat).cuda(),
                 'offset': torch.tensor([stop-start], dtype=torch.long, device='cuda')}
        torch.cuda.synchronize(); begin = time.perf_counter()
        with torch.no_grad():
            encoded = encoder(point)
            features, inverses = frozen_hierarchy(encoded)
        torch.cuda.synchronize(); encoder_seconds += time.perf_counter() - begin
        channels = [int(x.shape[1]) for x in features]
        payload = {'features': [x.cpu().half() for x in features],
                   'inverses': [x.cpu().int() for x in inverses],
                   'labels': torch.from_numpy(labels[selected].astype(np.uint8)),
                   'channels': channels, 'cache_id': cache_id, 'scene': scene, 'start': start, 'stop': stop}
        partial = target.with_suffix('.pth.part')
        storage_guard(8_000_000)
        torch.save(payload, partial); os.replace(partial, target)
        heartbeat(stage='scene_encode', scene=scene, block=block_number+1,
                  blocks=(voxel_count+block_size-1)//block_size)
        del encoded, features, inverses, point, payload
    result = {'cache_id': cache_id, 'source_identity': identity, 'array_sha256': hashes,
              'scene': scene, 'raw_points': raw_points, 'voxel_points': voxel_count, 'blocks': blocks,
              'channels': channels, 'read_seconds': load_seconds, 'voxel_seconds': voxel_seconds,
              'encoder_seconds': encoder_seconds, 'grid_size_native_units': grid_size,
              'inverse_complete': len(inverse) == raw_points and int(inverse.max()) < voxel_count}
    if not result['inverse_complete']: raise ValueError('Incomplete inverse mapping')
    atomic_json(complete, result)
    return result


def scene_predictions(folder, head):
    import torch
    folder = Path(folder)
    meta = json.loads((folder / 'SCENE_COMPLETE.json').read_text())
    prediction = np.empty(meta['voxel_points'], dtype=np.int16)
    head.eval()
    with torch.no_grad():
        for name in meta['blocks']:
            block = torch.load(folder/name, weights_only=True, map_location='cpu')
            features = [x.float().cuda() for x in block['features']]
            inverses = [x.long().cuda() for x in block['inverses']]
            logits = head(features, inverses)
            if not torch.isfinite(logits).all(): raise ValueError('Nonfinite evaluation logits')
            prediction[block['start']:block['stop']] = logits.argmax(-1).cpu().numpy()
    return prediction, meta


def evaluate_split(raw_root, scenes, cache_root, head, heartbeat):
    from metrics import confusion_matrix, evaluate
    counts = np.zeros((18, 18), dtype=np.int64)
    per_scene = []
    for index, scene in enumerate(scenes):
        heartbeat(stage='evaluate', scene=scene, scene_index=index, scene_count=len(scenes))
        prediction, meta = scene_predictions(Path(cache_root)/scene, head)
        for name, identity in meta['source_identity'].items():
            stat = (Path(raw_root)/scene/name).stat()
            if stat.st_size != identity['bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                raise ValueError('Raw source changed after cache creation')
        inverse = np.load(Path(cache_root)/scene/'inverse.npy', mmap_mode='r')
        labels = np.load(Path(cache_root)/scene/'labels.npy', mmap_mode='r')
        if len(labels) != meta['raw_points'] or len(inverse) != len(labels):
            raise ValueError('Raw-point coverage mismatch')
        scene_counts = np.zeros((18, 18), dtype=np.int64)
        for start in range(0, len(labels), 1_000_000):
            end = min(start+1_000_000, len(labels))
            scene_counts += confusion_matrix(labels[start:end], prediction[inverse[start:end]])
        if int(scene_counts.sum()) != len(labels): raise ValueError('Lost evaluation points')
        counts += scene_counts
        per_scene.append({'scene': scene, 'points': len(labels), 'confusion18': scene_counts.tolist()})
    result = evaluate(counts)
    result.update(scene_count=len(scenes), coverage=1., per_scene=per_scene)
    return result
