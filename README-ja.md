# TensorFunctions.jl

テンソル積の計算を簡単に書くためのヤツ

## なぜ作ったか

同じことをするパッケージに[TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)があるけど、微妙に書き方が気に入らないので作りました。裏ではTensorOperations.jlのfunction APIを使ってます。

## インストール

```julia
]add https://github.com/ho-oto/TensorFunctions.jl
```

## 使い方

```julia
A = randn(10,10); B = randn(10,10)
foo(a,b) = @tensorfunc [:a,:c] <= a[:a,:b] * b[:b,:c] # foo(A,B) == A*B
A = randn(10,10); B = randn(100)
bar(a,b) = @tensorfunc a[:a,:b] * b[(:b,:c)] => [:a,:c] # bar(A,B) == A * reshape(B,10,10)
A = randn(10,10); B = randn(200)
foofoo(a,b) = @tensorfunc [:a,:c,:d] <= a[:a,:b] * b[(:b,:c,:d|2)] # 推定できないボンド次元は顕に指定する必要あり
A = randn(10,10); B = randn(10,10); C = randn(10,10); D = randn(10,10)
barbar(a,b,c,d) = @tensorfunc (:d,:c,:b) [:a,:d] <= a[:a,:b] * b[:b,:c] * c[:c,:d] * d[:d,:e] # :d -> :c -> :b の順に縮約 (barbar(A,B,C,D) == A*(B*(C*D)))
A = randn(5,5,5,5,5); B = randn(5,5,5,5)
hoge(a,b) = @tensofunc [:a] <= a[:a,:b*4] * b[:b*4] # a[:a,:b1,:b2,:b3,:b4] * b[:b1,:b2,:b3,:b4] と書くのと同じ
```

## ドキュメント

準備中...?
