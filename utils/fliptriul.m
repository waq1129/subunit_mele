function A1 = fliptriul(A)
% a b c d     k i g d
% e f g 0 --> h f c 0
% h i 0 0     e b 0 0
% k 0 0 0     a 0 0 0

AA = fliplr(A);
AA = fliptriur(AA);
A1 = fliplr(AA);
