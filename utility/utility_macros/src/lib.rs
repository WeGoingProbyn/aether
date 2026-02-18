use proc_macro::TokenStream;
use quote::quote;
use syn::parse_quote;

#[proc_macro_attribute]
pub fn profile(_: TokenStream, item: TokenStream) -> TokenStream {
  // Try parsing as a free function first
  if let Ok(mut func) = syn::parse::<syn::ItemFn>(item.clone()) {
    let name = func.sig.ident.to_string();
    let guard: syn::Stmt = parse_quote! {
      let _guard = ::utility::profiler::SpanGuard::new(#name, module_path!());
    };
    func.block.stmts.insert(0, guard);
    return quote!(#func).into();
  }

  // Try parsing as an impl method
  if let Ok(mut method) = syn::parse::<syn::ImplItemFn>(item.clone()) {
    let name = method.sig.ident.to_string();
    let guard: syn::Stmt = parse_quote! {
      let _guard = ::utility::profiler::SpanGuard::new(#name, module_path!());
    };
    method.block.stmts.insert(0, guard);
    return quote!(#method).into();
  }

  item
}

#[proc_macro_derive(Serialize)]
pub fn serialize(input: TokenStream) -> TokenStream {
  let input = syn::parse::<syn::DeriveInput>(input).unwrap();
  let name = &input.ident;
  // Extract the named fields
  let fields = match &input.data {
    syn::Data::Struct(data) => match &data.fields {
      syn::Fields::Named(f) => &f.named,
      _ => panic!("only named structs supported"),
    },
    _ => panic!("only structs supported"),
  };

  let num_fields = fields.len();
  // Generate a serialize_struct_field call for each field
  let field_calls = fields.iter().map(|f| {
    let field_name = f.ident.as_ref().unwrap();
    let field_str = field_name.to_string();
    quote! {
      s.serialize_struct_field(#field_str, &self.#field_name)?;
    }
  });

  quote! {
    impl ::utility::serial::serialize::Serialize for #name {
      fn serialize<S: ::utility::serial::serialize::Serializer>(
        &self, s: &mut S
      ) -> Result<(), S::Error> {
        s.serialize_struct_begin(stringify!(#name), #num_fields)?;
        #(#field_calls)*
        s.serialize_struct_end()
      }
    }
  }
  .into()
}

#[proc_macro_derive(Deserialize)]
pub fn deserialize(input: TokenStream) -> TokenStream {
  let input = syn::parse::<syn::DeriveInput>(input).unwrap();
  let name = &input.ident;

  match &input.data {
    syn::Data::Struct(data) => match &data.fields {
      syn::Fields::Named(fields) => {
        let field_deserializations = fields.named.iter().map(|f| {
          let field_name = f.ident.as_ref().unwrap();
          let field_str = field_name.to_string();
          quote! {
            let #field_name = d.deserialize_struct_field(#field_str)?;
          }
        });

        let field_names = fields.named.iter().map(|f| {
          let field_name = f.ident.as_ref().unwrap();
          quote! { #field_name }
        });

        quote! {
          impl ::utility::serial::deserialize::Deserialize for #name {
            fn deserialize<D: ::utility::serial::deserialize::Deserializer>(
              d: &mut D
            ) -> Result<Self, D::Error> {
              d.deserialize_struct_begin(stringify!(#name))?;
              #(#field_deserializations)*
              d.deserialize_struct_end()?;
              Ok(Self { #(#field_names),* })
            }
          }
        }
        .into()
      }
      _ => panic!("Deserialize derive only supports named structs"),
    },
    _ => panic!("Deserialize derive only supports structs"),
  }
}
