import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import resized_crop, to_pil_image
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
def visualize_boxes(image, object_bounding_boxes, edge_bounding_boxes, labels, pos_losses, display_size=64):
    display_images = []

    for bb in object_bounding_boxes:
        new_image = torch.zeros(display_size, display_size, 3).cpu()
        
        x1,y1,x2,y2 = bb.to(torch.int)
        x1 = x1.item()
        x2 = x2.item()
        y1 = y1.item()
        y2 = y2.item()

        size = (x2-x1)*(y2-y1)

        if size > 10:
            new_image = resized_crop(image, y1, x1, y2-y1, x2-x1, (display_size, display_size))
            new_image = new_image.permute(1,2,0).cpu()/255
        
        display_images.append(new_image)

    display_images = torch.concat(display_images, dim=1)
    

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(display_images)
    ax[1].imshow(labels.detach().cpu().T)
    # Show all ticks and label them with the respective list entries
    # ax[1].set_xticks(np.arange(len(farmers)), labels=farmers)
    ax[1].set_yticks(torch.arange(labels.shape[1]), labels=["not visible", "visible", "ground truth not visible", "ground truth visible"])

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            text = ax[1].text(i, j, "{:.2f}".format(labels[i, j].item()),
                        ha="center", va="center", color="w")
            
    plt.savefig("nodes.png")

    fig, ax = plt.subplots(1, 2)
    edge_display = torch.zeros((display_size*len(object_bounding_boxes), display_size*len(object_bounding_boxes), 3))
    loss_display = torch.zeros((len(object_bounding_boxes), len(object_bounding_boxes), 1))

    edge_idx = 0
    for i in range(len(object_bounding_boxes)):
        for j in range(len(object_bounding_boxes)):
            if i == j:
                continue

            bbox = edge_bounding_boxes[edge_idx]

            x1,y1,x2,y2 = bbox.to(torch.int)
            x1 = x1.item()
            x2 = x2.item()
            y1 = y1.item()
            y2 = y2.item()

            size = (x2-x1)*(y2-y1)

            if size > 10:
                new_image = resized_crop(image, y1, x1, y2-y1, x2-x1, (display_size, display_size))
                new_image = new_image.permute(1,2,0).cpu()/255
                edge_display[i*display_size:(i+1)*display_size, j*display_size:(j+1)*display_size, :] = new_image
            loss_display[i,j] = pos_losses[edge_idx]

            edge_idx += 1

    ax[0].imshow(edge_display.detach().cpu())
    ax[1].imshow(loss_display.detach().cpu())
    # Show all ticks and label them with the respective list entries
    # ax[1].set_xticks(np.arange(len(farmers)), labels=farmers)
    # ax[1].set_yticks(torch.arange(labels.shape[1]), labels=["not visible", "visible", "ground truth not visible", "ground truth visible"])

    edge_idx = 0
    for i in range(len(object_bounding_boxes)):
        for j in range(len(object_bounding_boxes)):
            if i == j:
                continue

            text = ax[1].text(i, j, "{:.2f}".format(pos_losses[edge_idx].item()),
                    ha="center", va="center", color="w")
            edge_idx += 1
    plt.savefig("edges.png")

def visualize_boxes_pretrain(image, object_bounding_boxes, edge_bounding_boxes, labels, pos_losses, display_size=64):
    display_images = []

    for bb in object_bounding_boxes:
        new_image = torch.zeros(display_size, display_size, 3).cpu()
        
        x1,y1,x2,y2 = bb.to(torch.int)
        x1 = x1.item()
        x2 = x2.item()
        y1 = y1.item()
        y2 = y2.item()

        size = (x2-x1)*(y2-y1)

        if size > 10:
            new_image = resized_crop(image, y1, x1, y2-y1, x2-x1, (display_size, display_size))
            new_image = new_image.permute(1,2,0).cpu()/255
        
        display_images.append(new_image)

    display_images = torch.concat(display_images, dim=1)
    

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(display_images)
    ax[1].imshow(labels.detach().cpu().T)
    # Show all ticks and label them with the respective list entries
    # ax[1].set_xticks(np.arange(len(farmers)), labels=farmers)
    ax[1].set_yticks(torch.arange(labels.shape[1]), labels=["not visible", "visible", "ground truth not visible", "ground truth visible"])

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            text = ax[1].text(i, j, "{:.2f}".format(labels[i, j].item()),
                        ha="center", va="center", color="w")
            
    plt.savefig("nodes.png")

    fig, ax = plt.subplots(1, 2)
    edge_display = torch.zeros((display_size*len(object_bounding_boxes), display_size*len(object_bounding_boxes), 3))
    loss_display = torch.zeros((len(object_bounding_boxes), len(object_bounding_boxes), 1))

    edge_idx = 0
    for i in range(len(object_bounding_boxes)):
        for j in range(len(object_bounding_boxes)):
            if i == j:
                continue

            bbox = edge_bounding_boxes[edge_idx]

            x1,y1,x2,y2 = bbox.to(torch.int)
            x1 = x1.item()
            x2 = x2.item()
            y1 = y1.item()
            y2 = y2.item()

            size = (x2-x1)*(y2-y1)

            if size > 10:
                new_image = resized_crop(image, y1, x1, y2-y1, x2-x1, (display_size, display_size))
                new_image = new_image.permute(1,2,0).cpu()/255
                edge_display[i*display_size:(i+1)*display_size, j*display_size:(j+1)*display_size, :] = new_image
            loss_display[i,j] = pos_losses[edge_idx]

            edge_idx += 1

    ax[0].imshow(edge_display.detach().cpu())
    ax[1].imshow(loss_display.detach().cpu())
    # Show all ticks and label them with the respective list entries
    # ax[1].set_xticks(np.arange(len(farmers)), labels=farmers)
    # ax[1].set_yticks(torch.arange(labels.shape[1]), labels=["not visible", "visible", "ground truth not visible", "ground truth visible"])

    edge_idx = 0
    for i in range(len(object_bounding_boxes)):
        for j in range(len(object_bounding_boxes)):
            if i == j:
                continue

            text = ax[1].text(i, j, "{:.2f}".format(pos_losses[edge_idx].item()),
                    ha="center", va="center", color="w")
            edge_idx += 1
    plt.savefig("edges.png")


def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)

def draw_image(img, boxes, pred_labels):
    pic = to_pil_image(img)
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = pred_labels[i]

        box = boxes[i]
        # box = [box[0]*img.shape[1], box[1]*img.shape[2], 
        #        box[2]*img.shape[1], box[3]*img.shape[2]]

        draw_single_box(pic, box, draw_info=info)
    return pic

            
    


