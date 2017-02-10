n1=np.random.randint(Y_train.shape[0])
mask=Y_train[n1,1,:]


# fit an ellipse to mask
ellipse=get_ellipse(mask)
(c,r),(b,a),alfa=ellipse # (c,r) center coordinate, (b,a) major and minor radius, alfa: rotation angle

## create a binary mask from the fitted ellipse
elipse_mask=np.zeros_like(mask)
cv2.ellipse(elipse_mask, ellipse, (255,0, 0), -1) # elipse mask 



## split ellipse into two circles or ellipses
# centers of two circles/ellipses
alfa=alfa-90.
# convert to radians
alfa=alfa*np.pi/180.
if alfa<=(np.pi/2):
    c1=(c-a/4.*np.cos(alfa))
    r1=(r-a/4.*np.sin(alfa))
    c2=(c+a/4.*np.cos(alfa))
    r2=(r+a/4.*np.sin(alfa))
else:
    c1=(c+a/4.*np.cos(alfa))
    r1=(r-a/4.*np.sin(alfa))
    c2=(c-a/4.*np.cos(alfa))
    r2=(r+a/4.*np.sin(alfa))

# coodiates of ellipses
ellipse1=((c1,r1),(b/2,a/2),alfa)
ellipse2=((c2,r2),(b/2,a/2),alfa)

# coordinates of circles
radius=int(a/4)
c1=int(c1)
c2=int(c2)
r1=int(r1)
r2=int(r2)

## split to two circles
two_circle_mask=np.zeros_like(mask)
cv2.circle(two_circle_mask,(c1,r1), radius, (255,0,255), -1)
cv2.circle(two_circle_mask,(c2,r2), radius, (255,0,255), -1)
cv2.ellipse(two_circle_mask, ellipse, (255,0, 255), 1)

## split to two ellipses
two_ellipse_mask=np.zeros_like(mask)
cv2.ellipse(two_ellipse_mask, ellipse1, (255,0, 0), -1)
cv2.ellipse(two_ellipse_mask, ellipse2, (255,0, 255), -1)

# plots
plt.figure(figsize=(10,5))
plt.subplot(141)
plt.imshow(mask,cmap='gray')

# mask: a 2D binary mask
plt.subplot(142)
#cv2.ellipse(mask, ellipse, (255,0, 255), -1)
plt.imshow(elipse_mask)

plt.subplot(143)
plt.imshow(two_circle_mask)

plt.subplot(144)
plt.imshow(two_ellipse_mask)
plt.show()
