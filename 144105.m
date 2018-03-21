


process1=im2uint8(zeros(250,250,3,5));


process2=im2uint8(zeros(250,250,3,5));


process3=im2uint8(zeros(250,250,3,5));

tayyab=VideoReader('Tayyab.mp4');

obj1=1;

for img1 = 1:10:50
    
    S_S= read(tayyab, img1);
   
    filename1=strcat('Tayyab_pic',num2str(img1),'.jpg');
    
    
    new_image=imrotate(S_S,90);
   
    FD = vision.CascadeObjectDetector;
    
   
    BB = step(FD,new_image);
  
    for i = 1:size(BB,1)
        rectangle('Position',BB(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
    end

    
    
    for l = 1:size(BB,1)
        aa= imcrop(new_image,BB(l,:));
        aa=imresize(aa,[250 250]);
        imwrite(aa,filename1);
    end
     process1(:,:,:,obj1)=aa;
   obj1=obj1+1;
    
end

% for a=1:1:5
%     imshow(process1(:,:,:,a));
% end


obj1=1;
Muaaz=VideoReader('Muaaz.mp4');

bb=1;

for img2 = 1:10:100
    S_S2= read(Muaaz, img2);
    filename2=strcat('Muaaz_pic',num2str(img2),'.jpg');
    
    
    new_image2=imrotate(S_S2,90);
    

    
    FDetect1 = vision.CascadeObjectDetector;
    

    
    BB1 = step(FDetect1,new_image2);
    
    for g = 1:size(BB1,1)
        rectangle('Position',BB1(g,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
    end
    
    
    
    
    for k = 1:size(BB1,1)
        bb= imcrop(new_image2,BB1(k,:));
        bb=imresize(bb,[250 250]);
        imwrite(bb,filename2);
    end
     process2(:,:,:,obj1)=bb;
   obj1=obj1+1;
     
end


for b=1:1:5
    imshow(process2(:,:,:,b));
end
    

obj1=1;



% Mansoor=VideoReader('Mansoor.mp4');
% 
% crp=1;
% 
% for img3=25:25:225
%     S_S3= read(Mansoor, img3);
%     filename3=strcat('Mansoor_pic',num2str(img3),'.jpg');
%     
%     new_image3=imrotate(S_S3,90);
%     imwrite(new_image3,filename3);
%   
%     FDetect2 = vision.CascadeObjectDetector;
%     
%    
%     BB2 = step(FDetect2,new_image3);
%     
%     for h = 1:size(BB2,1)
%         rectangle('Position',BB2(h,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
%     end
%     
%     
%     
%     
%     for a = 1:size(BB2,1)
%         
%         crp= imcrop(new_image3,BB2(a,:));
%         
%         crp=imresize(crp,[250 250]);
%         
%         imwrite(crp,filename3);
%     end
%      process3(:,:,:,obj1)=crp;
%    obj1=obj1+1;
%     
% end

% 
% for a=1:1:5
%     imshow(process3(:,:,:,a));
% end




           % now iam getting Features of trained data
           

           MyFeature_matrix=zeros(15,32400);

           index=1;

           i=1;

while i<=5
      
    MyFeature_matrix(i,:)=extractHOGFeatures(process1(:,:,:,i));
    
    i=i+1; 
end



while i<=10
        
    MyFeature_matrix(i,:)=extractHOGFeatures(process2(:,:,:,index));
       
        i=i+1;
        
        index=index+1;
end

index=1;
while i<=15
      
    MyFeature_matrix(i,:)=extractHOGFeatures(process3(:,:,:,index));
    
    i=i+1;
    
    index=index+1;
end




LABEL=zeros(15,1);           %% Creating Label Matrix for 3 persons




for i=1:1:5

    LABEL(i)=1;
end

for i=6:1:10

    LABEL(i)=2;
end

for i=10:1:15

    LABEL(i)=3;
end






%% Using a sample and extracting it features



temporary=imread('abc.jpg');

FD = vision.CascadeObjectDetector;

length = step(FD,temporary);

process4=im2uint8(zeros(250,250,3,size(length,1)));

for i = 1:size(length,1)
    
    rectangle('Position',length(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');

end

    for i = 1:size(length,1)
        
     mytestimage= imcrop(temporary,length(i,:));
     
     mytestimage=imresize(mytestimage,[250 250]);
     
     process4(:,:,:,i)=mytestimage;
     
    end
    
    
    
    
%%%%% NOW checking it with sample data

temp1=0;

temp2=0;

temp3=0;

Testing=zeros(size(length,1),32400);

for i=1:size(length,1)
    
Testing(i,:)=extractHOGFeatures(process4(:,:,:,i));

end

abc=fitcecoc(MyFeature_matrix,LABEL);

LABEL2=zeros(size(length,1),1);

for imgg=1:1:size(length,1)
    
LABEL2(imgg,1)=predict(abc,Testing(imgg,:));

end

FD = vision.CascadeObjectDetector;

A = step(FD,temporary);



for i = 1:size(A,1)
    
    rectangle('Position',A(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
    
end

for w=1:1:size(A,1)
    
if LABEL2(w,1)==1

    alpha=insertObjectAnnotation(temporary,'Rectangle',A(w,:),'tayyab','FontSize',40);

    temp1=1;
end

if LABEL2(w,1)==2

    beta=insertObjectAnnotation(temporary,'Rectangle',A(w,:),'muaaz','FontSize',40);

    temp2=1;
end

if LABEL2(w,1)==3

    charlie=insertObjectAnnotation(temporary,'Rectangle',A(w,:),'mansoor','FontSize',40);

    temp3=1;
end


end

if temp2==0 
    if temp3==0
    
        imshow(alpha);
    
    end
end


if temp1==0 
   
    if temp3==0
    
        imshow(beta);
    
    end
end
if temp1==0 
   
    if temp2==0
    
        imshow(charlie);
   
    end
end

if temp3==0 
    if temp1~=0 
        if temp2 ~=0
  
            delta=imadd(alpha,beta);
           
            imshow(delta);
        end
    end
end

if temp2==0 
    if  temp1~=0 
        if temp3~=0
    
            delta=imadd(alpha,charlie);
    
            imshow(delta);
        end
    end
end

if temp1==0  
    if temp2~=0 
        if temp3~=0
    
            delta=imadd(beta,charlie);
    
            imshow(delta);
        end
    end
end


if temp1~=0 
    if temp2~=0 
        if temp3~=0
   
            delta=imadd(alpha,beta);
   
            extra=imadd(delta,charlie);
    
            imshow(extra);
        end
    end
end






% label=predict(model,Testing(1,:));
% 
% 
% FD = vision.CascadeObjectDetector;
% 
% length = step(FD,temporary);
% 
% for i = 1:size(length,1)
%     rectangle('Position',length(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
% end
% 
% if label==1
% b=insertObjectAnnotation(temporary,'Rectangle',length,'tayyab','FontSize',30);
% end
% if label==2
% b=insertObjectAnnotation(temporary,'Rectangle',length,'muaaz');
% end
% % if label==3
% % b=insertObjectAnnotation(temporary,'Rectangle',length,'hamzaMansoor');
% % end
% 
% imshow(b);
 







