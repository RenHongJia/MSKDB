function DAG = learn_struct_mskdb(data, class_node, node_sizes,k)
% learn_struct_mskdb 
% (with discrete nodes)
% dag = learn_struct_mskdb(data, class_node, node_sizes,k)
% Input :
% 	data(i,m) is the value of node i in case m；节点 × 样本
% 	class_node is the class node
%   node_sizes 
%
% Output :
%	dag = adjacency matrix of the dag

if nargin == 3
    k=2;
end;

N=size(data,1);
dag=zeros(N);
notClass = setdiff(1:N,class_node);
[sort_attribute,CMI,~] = attrbute_sort(data,node_sizes,class_node);


DAG= learn_struct(dag,sort_attribute,CMI,class_node,k);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sort_attribute,CMI,MI_xc] =   attrbute_sort(data,node_sizes,class_node)

Nnode = size(data,1);
notClass = setdiff(1:Nnode,class_node);

CMI = zeros(Nnode,Nnode); % CMI(x,y|c)
MI_xc = zeros(1,length(notClass)); % MI(x,c)
H = zeros(1,Nnode); % H(x)
MI_Ratio = zeros(1,length(notClass)); % MI_xc/H(x)

for entropy_i = 1: Nnode 
    
    H(entropy_i) = entropy_ren(entropy_i ,data,node_sizes(entropy_i)); % 节点的熵
    if entropy_i~=class_node
        laplas=0;
        MI_xc(entropy_i) = mutual_info_score_ren(data,entropy_i ,class_node ,laplas ,node_sizes(entropy_i) ,node_sizes(class_node)); % 属性与类节点之间的互信息
        MI_Ratio(entropy_i) =  MI_xc(entropy_i)/H(entropy_i);
    end
    for entropy_j = 1: Nnode % size(notclass,2)
        if entropy_i~=entropy_j && entropy_j~=class_node && entropy_i~=class_node
            CMI(entropy_i,entropy_j) = cond_mutual_info_score_ren(data,entropy_i,entropy_j,class_node,node_sizes(entropy_i),node_sizes(entropy_j),node_sizes(class_node));
        end
    end
end
[~, MI_Ratio_I] = sort(MI_Ratio,'descend');
sort_attribute =MI_Ratio_I;
end

function [dag]= learn_struct(dag,sort_att,CMI,class_node,K)

for  i =1:length(sort_att)    
    if i ==1 
        dag(class_node,sort_att(i))=1;
       
    else 
        if i<= (K+1)  
            dag(class_node,sort_att(i))=1;
            for  j =1:i-1 
                dag(sort_att(j),sort_att(i))=1;
            end
        else  
       
            for  j =1:i-1
                temp(j,1) = sort_att(j); 
                temp(j,2) = CMI(sort_att(i),sort_att(j));  
            end            
                     
            [~,I]=sort(temp(:,2),'descend');
            A=temp(I,:);
            parent_set = A(1:K,1);
            dag(class_node,sort_att(i))=1;
            dag(parent_set,sort_att(i))=1;
                              
            clear temp;
        end
    end
end
end




