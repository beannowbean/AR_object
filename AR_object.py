import cv2
import numpy as np

# --- 1. 캘리브레이션을 위한 준비 ---
# 체커보드 교차점의 termination criteria (정밀도 및 반복횟수)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체커보드 3D 좌표 준비 (7x7 교차점, 각 점을 1단위 간격으로 설정)
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

# 캘리브레이션 시 사용할 3D, 2D 좌표 배열
objpoints = []  # 실세계 3D 점들
imgpoints = []  # 영상상의 2D 점들

# video.mp4 파일 읽기 (체커보드 영상)
cap = cv2.VideoCapture('video.mp4')
calib_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 체커보드 코너 검출 (내부 코너가 7x7개)
    ret_cb, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    if ret_cb:
        calib_frame_count += 1
        objpoints.append(objp)
        # 코너 위치를 더욱 정밀하게 보정
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # 검출된 코너를 그림으로 표시 (디버깅용)
        cv2.drawChessboardCorners(frame, (7, 7), corners2, ret_cb)
        cv2.imshow('Calibration', frame)
        cv2.waitKey(100)
    # 충분한 프레임 (예제에서는 20프레임) 후 종료
    if calib_frame_count >= 20:
        break

cap.release()
cv2.destroyAllWindows()

# --- 2. 카메라 캘리브레이션 수행 ---
ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Calibration done. RMS error:", ret_calib)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# --- 3. AR 증강: 3D 초록색 정육면체 오버레이 ---
# 정육면체의 8개 꼭지점 (한 변의 길이를 1 단위로 사용; 체커보드의 한 칸 크기로 가정)
cube_points = np.float32([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, -1],
    [1, 1, -1],
    [0, 1, -1]
])

# 각 면을 이루는 꼭지점 인덱스 (여섯 면: 밑, 앞, 오른쪽, 뒤, 왼쪽, 위)
faces = [
    [0, 1, 2, 3],  # 밑면 (체커보드 위에 닿아 있음)
    [0, 1, 5, 4],  # 앞면
    [1, 2, 6, 5],  # 오른쪽 면
    [2, 3, 7, 6],  # 뒷면
    [3, 0, 4, 7],  # 왼쪽 면
    [4, 5, 6, 7]   # 윗면
]

# face_depth: 각 면의 평균 깊이(카메라 좌표계에서의 z값)를 계산하는 함수
def face_depth(face, rvec, tvec, cube_points):
    R, _ = cv2.Rodrigues(rvec)
    depths = []
    for idx in face:
        point_3d = cube_points[idx].reshape(3, 1)
        point_cam = np.dot(R, point_3d) + tvec  # 카메라 좌표계로 변환
        depths.append(point_cam[2][0])
    return np.mean(depths)

# AR 오버레이를 위한 비디오 재생
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    if ret_cb:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # cv2.solvePnP를 사용하여 체커보드의 자세(회전, 이동)를 계산
        ret_solve, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
        # 3D 정육면체의 꼭지점들을 영상상에 투영
        imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, mtx, dist)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # 오버레이를 위한 복사본 생성
        frame_aug = frame.copy()

        # 각 면의 깊이를 계산한 후, 깊이가 먼 면부터 그리기 (painter's algorithm)
        faces_depth = []
        for face in faces:
            depth = face_depth(face, rvec, tvec, cube_points)
            faces_depth.append((depth, face))
        faces_depth.sort(key=lambda x: x[0], reverse=True)  # 멀리 있는 면부터

        # 각 면을 채워서 그리기 (6면 모두 초록색으로 채움)
        for depth, face in faces_depth:
            pts = imgpts[face]
            cv2.fillConvexPoly(frame_aug, pts, (0, 255, 0))  # 초록색 채우기
            cv2.polylines(frame_aug, [pts], True, (0, 0, 0), 2)  # 외곽선 그리기

        cv2.imshow('AR Cube', frame_aug)
    else:
        cv2.imshow('AR Cube', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
