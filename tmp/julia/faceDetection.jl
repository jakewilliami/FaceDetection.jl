#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/src/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#

include("cv2.jl")

function detect(img::cv2.Image, cascade)
    rects = cv2.CascadeClassifier_detectMultiScale(cascade, img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for ind in 1:size(rects, 1)
        rects[ind] = (rects[ind][1], rects[ind][2], rects[ind][3]+rects[ind][1], rects[ind][4]+rects[ind][2])
    end
    return rects
end

function draw_rects(img, rects, color, offset=(0,0))
    for x in rects
        cv2.rectangle(img, (offset[1]+x[1], offset[2]+x[2]), (offset[1]+x[3], offset[2]+x[4]), color, thickness = 2)
    end
end

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
nested = cv2.CascadeClassifier("haarcascade_eye.xml")

while true
    ret, img = cv2.VideoCapture_read(cap)
    if ret==false
        break
    end
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    rects = detect(gray, cascade)
    vis = copy(img)
    draw_rects(vis, rects, (0.0, 255.0, 0.0))

    if ~cv2.CascadeClassifier_empty(nested)
        for x in rects
            roi = view(gray, :, Int64(x[1]):Int64(x[3]), Int64(x[2]):Int64(x[4]))
            subrects = detect(roi, nested)
            draw_view = view(vis, :, Int64(x[1]):Int64(x[3]), Int64(x[2]):Int64(x[4]))
            draw_rects(draw_view, subrects, (255.0, 0.0, 0.0))
        end
    end

    cv2.imshow("facedetect", vis)
    if cv2.waitKey(5)==27
        break
    end
end

cv2.destroyAllWindows()