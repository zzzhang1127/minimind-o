"""
Custom collate function for OLM dataset that properly handles variable-length speech
"""
import torch
import torch.nn.functional as F


def olm_collate_fn(batch):
    """
    Collate function for OLMDataset that:
    1. Handles variable-length speech tensors
    2. Pads them to the same length in batch
    3. Creates attention masks for speech
    4. Pads input_ids and labels properly
    """
    # Unpack batch
    input_ids_list = []
    labels_list = []
    pixel_values_list = []
    speech_values_list = []
    speech_lengths_original = []
    
    max_speech_len = 0
    max_input_len = 0
    
    for item in batch:
        input_ids, labels, pixel_values, speech_values, speech_lengths = item
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        pixel_values_list.append(pixel_values)
        speech_values_list.append(speech_values)
        speech_lengths_original.append(speech_lengths)
        
        # Track max lengths for padding
        max_input_len = max(max_input_len, input_ids.shape[0])
        # speech_values shape: (1, time_steps, n_mels)
        max_speech_len = max(max_speech_len, speech_values.shape[1])
    
    # Pad input_ids to same length
    padded_input_ids = []
    for input_ids in input_ids_list:
        if input_ids.shape[0] < max_input_len:
            padding = torch.full((max_input_len - input_ids.shape[0],), 0, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding])
        padded_input_ids.append(input_ids)
    input_ids_batch = torch.stack(padded_input_ids)
    
    # Pad labels to same length
    padded_labels = []
    for labels in labels_list:
        if labels.shape[0] < max_input_len:
            padding = torch.full((max_input_len - labels.shape[0],), -100, dtype=labels.dtype)
            labels = torch.cat([labels, padding])
        padded_labels.append(labels)
    labels_batch = torch.stack(padded_labels)
    
    # Pad speech to same length
    # speech_values shape: (1, time_steps, n_mels) or (1, 1, time_steps, n_mels)
    padded_speech = []
    speech_mask_list = []
    
    for speech_values in speech_values_list:
        # Ensure speech_values is 4D: (1, 1, time_steps, n_mels)
        if len(speech_values.shape) == 3:
            speech_values = speech_values.unsqueeze(1)
        
        bs, num_speech, time_steps, n_mels = speech_values.shape
        
        # Pad time dimension
        if time_steps < max_speech_len:
            padding = (0, 0, 0, max_speech_len - time_steps)
            speech_values = F.pad(speech_values, padding, mode='constant', value=0)
        
        padded_speech.append(speech_values)
        
        # Create attention mask: 1 for valid frames, 0 for padded frames
        speech_mask = torch.ones((bs, num_speech, max_speech_len), dtype=torch.long)
        if time_steps < max_speech_len:
            speech_mask[:, :, time_steps:] = 0
        speech_mask_list.append(speech_mask)
    
    speech_values_batch = torch.cat(padded_speech, dim=0)  # Concatenate batch dim
    speech_mask_batch = torch.cat(speech_mask_list, dim=0)
    
    # Stack pixel values (usually already same size)
    pixel_values_batch = torch.cat(pixel_values_list, dim=0) if pixel_values_list[0] is not None else None
    
    # Keep original speech lengths
    speech_lengths_batch = torch.stack(speech_lengths_original)
    
    return (
        input_ids_batch,
        labels_batch,
        pixel_values_batch,
        speech_values_batch,
        speech_lengths_batch,
        speech_mask_batch,  # New: attention mask for speech
    )
