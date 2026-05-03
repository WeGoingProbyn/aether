// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

mod diagnostics;

use proc_macro::TokenStream;
use quote::quote;
use syn::{LitStr, parse_quote};

#[proc_macro_derive(StateDiagnostics, attributes(diagnostics))]
pub fn state_diagnostics(input: TokenStream) -> TokenStream {
  diagnostics::expand(input)
}

#[proc_macro_attribute]
pub fn profile(attr: TokenStream, item: TokenStream) -> TokenStream {
  let explicit_name = match profile_attr_name(attr) {
    Ok(name) => name,
    Err(err) => return err,
  };

  // Try parsing as a free function first
  if let Ok(mut func) = syn::parse::<syn::ItemFn>(item.clone()) {
    let name = explicit_name.unwrap_or_else(|| func.sig.ident.to_string());
    func.block.stmts.insert(0, profile_guard(name));
    return quote!(#func).into();
  }

  // Try parsing as an impl method
  if let Ok(mut method) = syn::parse::<syn::ImplItemFn>(item.clone()) {
    let name = explicit_name.unwrap_or_else(|| method.sig.ident.to_string());
    method.block.stmts.insert(0, profile_guard(name));
    return quote!(#method).into();
  }

  item
}

fn profile_attr_name(attr: TokenStream) -> Result<Option<String>, TokenStream> {
  if attr.is_empty() {
    return Ok(None);
  }

  syn::parse::<LitStr>(attr)
    .map(|lit| Some(lit.value()))
    .map_err(|err| err.to_compile_error().into())
}

fn profile_guard(name: String) -> syn::Stmt {
  parse_quote! {
    let _guard = ::utility::profiler::SpanGuard::new(#name, module_path!());
  }
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
